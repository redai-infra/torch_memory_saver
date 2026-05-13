#include "core.h"
#include "utils.h"
#include "macro.h"
#include "api_forwarder.h"

// When TMS_DISABLE_PINNED_CPU_BACKUP=1, skip cudaMallocHost entirely and use
// plain std::malloc for all CPU backup buffers.  This avoids exhausting the
// CUDA driver's pinned-memory quota on large-model clusters (e.g. 744B MoE)
// where hundreds of GB are offloaded per node, which can trigger cudaMallocHost
// failures and – in the worst case – host-OOM / raylet crashes.
static bool tms_disable_pinned_cpu_backup() {
    static bool val = get_bool_env_var("TMS_DISABLE_PINNED_CPU_BACKUP");
    return val;
}

// When TMS_RELEASE_CPU_BACKUP_AFTER_RESUME=1, automatically release all
// CPU backup buffers after resume() restores data to GPU.  This trades
// re-allocation cost on the next pause() for immediate host-memory savings,
// which is critical for very large models (e.g. 744B MoE) where the
// combined footprint of GPU-resident weights + CPU backup + checkpoint
// serialisation can push the node past its RAM limit.
static bool tms_release_cpu_backup_after_resume() {
    static bool val = get_bool_env_var("TMS_RELEASE_CPU_BACKUP_AFTER_RESUME");
    return val;
}

TorchMemorySaver::TorchMemorySaver() {}

TorchMemorySaver &TorchMemorySaver::instance() {
    static TorchMemorySaver instance;
    return instance;
}

cudaError_t TorchMemorySaver::malloc(void **ptr, CUdevice device, size_t size, const std::string& tag, const bool enable_cpu_backup) {
#if TMS_ROCM_LEGACY_CHUNKED
    return ROCmHIPImplementation::rocm_malloc(ptr, device, size, tag, enable_cpu_backup, allocation_metadata_, allocator_metadata_mutex_);

#else
    const uint64_t memory_margin_bytes = memory_margin_bytes_.load();
    if (memory_margin_bytes > 0) {
        size_t free_bytes, total_bytes;
        CUDA_ERROR_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (memory_margin_bytes + size > free_bytes) {
            std::cout << "[torch_memory_saver.cpp] TorchMemorySaver::malloc return OOM since"
                << " memory_margin_bytes=" << memory_margin_bytes
                << " (alloc)size=" << size
                << " free_bytes=" << free_bytes
                << std::endl;
            return cudaErrorMemoryAllocation;
        }
    }

    CUmemGenericAllocationHandle allocHandle;

    cudaError_t ret = CUDAUtils::cu_mem_create(&allocHandle, size, device);
    if (ret != cudaSuccess) {
        return ret;
    }

    CURESULT_CHECK(cuMemAddressReserve((CUdeviceptr *) ptr, size, 0, 0, 0));
    CURESULT_CHECK(cuMemMap((CUdeviceptr) * ptr, size, 0, allocHandle, 0));
    CUDAUtils::cu_mem_set_access(*ptr, size, device);

    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        allocation_metadata_.emplace(
            *ptr,
            AllocationMetadata{size, device, tag, AllocationState::ACTIVE, enable_cpu_backup, nullptr, /*cpu_backup_is_pinned=*/true, allocHandle}
        );
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.malloc "
              << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size
              << " allocHandle=" << allocHandle << " tag=" << tag
              << std::endl;
#endif

#endif
    return cudaSuccess;
}

cudaError_t TorchMemorySaver::free(void *ptr) {
#if TMS_ROCM_LEGACY_CHUNKED
    return ROCmHIPImplementation::rocm_free(ptr, allocation_metadata_, allocator_metadata_mutex_);

#else
    AllocationMetadata metadata;
    {
        const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);
        if (allocation_metadata_.count(ptr) == 0) {
            return APIForwarder::call_real_cuda_free(ptr);
        }

        metadata = allocation_metadata_[ptr];
        allocation_metadata_.erase(ptr);
    }

    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
    CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
    CURESULT_CHECK(cuMemAddressFree((CUdeviceptr) ptr, metadata.size));

    if (nullptr != metadata.cpu_backup) {
        if (metadata.cpu_backup_is_pinned) {
            CUDA_ERROR_CHECK(cudaFreeHost(metadata.cpu_backup));
        } else {
            std::free(metadata.cpu_backup);
        }
        metadata.cpu_backup = nullptr;
    }

    TMS_LOG_DEBUG("TorchMemorySaver.free"
              << " ptr=" << ptr << " size=" << metadata.size
              << " allocHandle=" << metadata.allocHandle << " tag=" << metadata.tag);

#endif
    return cudaSuccess;
}

void TorchMemorySaver::pause(const std::string& tag) {
#if TMS_ROCM_LEGACY_CHUNKED
    ROCmHIPImplementation::rocm_pause(tag, allocation_metadata_, allocator_metadata_mutex_);

#else
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    // Counters for summary log
    size_t total_count = 0, total_bytes = 0;
    size_t cpu_backup_count = 0, cpu_backup_bytes = 0;
    size_t pinned_alloc_count = 0, unpinned_fallback_count = 0;
    size_t reused_backup_count = 0;
    size_t skipped_count = 0;

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata& metadata = it->second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }

        if (metadata.state != AllocationState::ACTIVE) {
            TMS_LOG_WARN("Cannot pause allocation that is not active."
                      << " tag=" << metadata.tag << " ptr=" << (uintptr_t)ptr
                      << " size=" << (metadata.size / 1024 / 1024) << "MB"
                      << " state=" << (int)metadata.state);
            skipped_count++;
            continue;  // Skip instead of exit(1) – other allocations may still be pauseable
        }

        total_count++;
        total_bytes += metadata.size;

        if (metadata.enable_cpu_backup) {
            cpu_backup_count++;
            cpu_backup_bytes += metadata.size;

            if (nullptr == metadata.cpu_backup) {
                bool use_pinned = !tms_disable_pinned_cpu_backup();
                bool pinned_ok = false;

                if (use_pinned) {
                    cudaError_t alloc_err = cudaMallocHost(&metadata.cpu_backup, metadata.size);
                    if (alloc_err != cudaSuccess) {
                        const char* err_str = cudaGetErrorString(alloc_err);
                        TMS_LOG_WARN("cudaMallocHost failed, falling back to unpinned malloc."
                                  << " tag=" << metadata.tag
                                  << " size=" << (metadata.size / 1024 / 1024) << "MB"
                                  << " err=" << alloc_err << " (" << (err_str ? err_str : "?") << ")");
                        cudaGetLastError();
                    } else {
                        pinned_ok = true;
                    }
                }

                if (!pinned_ok) {
                    metadata.cpu_backup = std::malloc(metadata.size);
                    if (metadata.cpu_backup == nullptr) {
                        TMS_LOG_ERROR("FATAL: malloc failed."
                                  << " tag=" << metadata.tag
                                  << " size=" << (metadata.size / 1024 / 1024) << "MB"
                                  << " total_host_ram_requested_so_far=" << (cpu_backup_bytes / 1024 / 1024) << "MB");
                        exit(1);
                    }
                    metadata.cpu_backup_is_pinned = false;
                    unpinned_fallback_count++;
                } else {
                    metadata.cpu_backup_is_pinned = true;
                    pinned_alloc_count++;
                }
            } else {
                // Reusing previously allocated cpu_backup (lazy-free optimisation)
                reused_backup_count++;
            }

            SIMPLE_CHECK(metadata.cpu_backup != nullptr, "cpu_backup should not be nullptr");
            // TODO may use cudaMemcpyAsync if needed
            CUDA_ERROR_CHECK(cudaMemcpy(metadata.cpu_backup, ptr, metadata.size, cudaMemcpyDeviceToHost));
        }

        // Use non-fatal versions so a single unmap/release failure doesn't kill the process
        CURESULT_WARN(cuMemUnmap((CUdeviceptr) ptr, metadata.size),
                      "tag=" << metadata.tag << " ptr=" << (uintptr_t)ptr
                      << " size=" << (metadata.size / 1024 / 1024) << "MB");
        CURESULT_WARN(cuMemRelease(metadata.allocHandle),
                      "tag=" << metadata.tag << " allocHandle=" << metadata.allocHandle
                      << " size=" << (metadata.size / 1024 / 1024) << "MB");

        metadata.state = AllocationState::PAUSED;

        TMS_LOG_DEBUG("pause alloc: ptr=" << (uintptr_t)ptr
                  << " size=" << (metadata.size / 1024 / 1024) << "MB"
                  << " tag=" << metadata.tag
                  << " cpu_backup=" << (metadata.enable_cpu_backup ? "yes" : "no")
                  << " pinned=" << (metadata.cpu_backup_is_pinned ? "yes" : "no"));
    }

    TMS_LOG_INFO("pause complete: filter_tag=" << (tag.empty() ? "(all)" : tag)
              << " total_allocs=" << total_count
              << " total_bytes=" << (total_bytes / 1024 / 1024) << "MB"
              << " cpu_backup=" << cpu_backup_count << "(" << (cpu_backup_bytes / 1024 / 1024) << "MB)"
              << " new_pinned=" << pinned_alloc_count
              << " new_unpinned_fallback=" << unpinned_fallback_count
              << " reused=" << reused_backup_count
              << " skipped_not_active=" << skipped_count);
#endif
}

void TorchMemorySaver::resume(const std::string& tag) {
#if TMS_ROCM_LEGACY_CHUNKED
    ROCmHIPImplementation::rocm_resume(tag, allocation_metadata_, allocator_metadata_mutex_);

#else
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    size_t total_count = 0, total_bytes = 0;
    size_t cpu_restore_count = 0;
    size_t skipped_count = 0;

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata &metadata = it->second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }

        if (metadata.state != AllocationState::PAUSED) {
            TMS_LOG_WARN("Cannot resume allocation that is not paused."
                      << " tag=" << metadata.tag << " ptr=" << (uintptr_t)ptr
                      << " size=" << (metadata.size / 1024 / 1024) << "MB"
                      << " state=" << (int)metadata.state);
            skipped_count++;
            continue;  // Skip instead of exit(1)
        }

        total_count++;
        total_bytes += metadata.size;

        CUmemGenericAllocationHandle newAllocHandle;
        CUDA_ERROR_CHECK(CUDAUtils::cu_mem_create(&newAllocHandle, metadata.size, metadata.device));

        CURESULT_CHECK(cuMemMap((CUdeviceptr) ptr, metadata.size, 0, newAllocHandle, 0));

        CUDAUtils::cu_mem_set_access(ptr, metadata.size, metadata.device);

        if (metadata.enable_cpu_backup) {
            SIMPLE_CHECK(metadata.cpu_backup != nullptr, "cpu_backup should not be nullptr");
            // TODO may use cudaMemcpyAsync if needed
            CUDA_ERROR_CHECK(cudaMemcpy(ptr, metadata.cpu_backup, metadata.size, cudaMemcpyHostToDevice));

            // Lazy free: keep cpu_backup alive to avoid re-allocation on next
            // pause() cycle.  The buffer will be freed when the allocation
            // itself is freed via TorchMemorySaver::free().
            cpu_restore_count++;
        }

        TMS_LOG_DEBUG("resume alloc: ptr=" << (uintptr_t)ptr
                  << " size=" << (metadata.size / 1024 / 1024) << "MB"
                  << " tag=" << metadata.tag
                  << " oldHandle=" << metadata.allocHandle
                  << " newHandle=" << newAllocHandle
                  << " cpu_backup=" << (metadata.enable_cpu_backup ? "yes" : "no"));

        metadata.state = AllocationState::ACTIVE;
        metadata.allocHandle = newAllocHandle;
    }

    TMS_LOG_INFO("resume complete: filter_tag=" << (tag.empty() ? "(all)" : tag)
              << " total_allocs=" << total_count
              << " total_bytes=" << (total_bytes / 1024 / 1024) << "MB"
              << " cpu_restored=" << cpu_restore_count
              << " skipped_not_paused=" << skipped_count);

    // Optionally release CPU backup buffers now that GPU data is restored.
    // This avoids holding duplicate copies (GPU + CPU) simultaneously,
    // which can cause host OOM on very large models during checkpoint saving.
    // Controlled by TMS_RELEASE_CPU_BACKUP_AFTER_RESUME=1.
    if (tms_release_cpu_backup_after_resume()) {
        size_t release_count = 0, release_bytes = 0;
        for (auto& [p, md] : allocation_metadata_) {
            if (!tag.empty() && md.tag != tag) continue;
            if (md.state != AllocationState::ACTIVE) continue;
            if (md.cpu_backup == nullptr) continue;

            if (md.cpu_backup_is_pinned) {
                cudaFreeHost(md.cpu_backup);
            } else {
                std::free(md.cpu_backup);
            }
            md.cpu_backup = nullptr;
            release_count++;
            release_bytes += md.size;
        }
        TMS_LOG_INFO("release_cpu_backup_after_resume: released=" << release_count
                  << " freed_bytes=" << (release_bytes / 1024 / 1024) << "MB");
    }
#endif
}

uint8_t* TorchMemorySaver::get_cpu_backup_pointer(const uint8_t* query_gpu_ptr, uint64_t query_size) {
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        uint8_t *ptr = (uint8_t*) it->first;
        AllocationMetadata &metadata = it->second;

#if TMS_ROCM_LEGACY_CHUNKED
        size_t total_size = metadata.aligned_size;
#else
        size_t total_size = metadata.size;
#endif

        if ((ptr <= query_gpu_ptr) && (query_gpu_ptr + query_size <= ptr + total_size)) {
            const size_t offset = query_gpu_ptr - ptr;
            if (metadata.state == AllocationState::ACTIVE) {
                return nullptr;
            } else {
                SIMPLE_CHECK(nullptr != metadata.cpu_backup,
                    "get_cpu_backup_pointer: found paused allocation but cpu_backup does not exist, do you forget to enable cpu backup");
                return (uint8_t*) metadata.cpu_backup + offset;
            }
        }
    }

    std::cerr << "[torch_memory_saver.cpp] get_cpu_backup_pointer fail to find backup "
              << " query_gpu_ptr=" << query_gpu_ptr << " query_size=" << query_size
              << std::endl;
    exit(1);
}

void TorchMemorySaver::pre_allocate_cpu_backup(const std::string& tag) {
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);

    size_t count = 0, total_bytes = 0;
    size_t pinned_count = 0, unpinned_count = 0, skipped_count = 0;

    for (auto& [ptr, metadata] : allocation_metadata_) {
        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }
        if (!metadata.enable_cpu_backup) {
            continue;
        }
        if (metadata.cpu_backup != nullptr) {
            // Already allocated (e.g. from a previous pre_allocate or pause cycle)
            skipped_count++;
            continue;
        }

        bool use_pinned = !tms_disable_pinned_cpu_backup();
        bool pinned_ok = false;

        if (use_pinned) {
            cudaError_t err = cudaMallocHost(&metadata.cpu_backup, metadata.size);
            if (err != cudaSuccess) {
                const char* err_str = cudaGetErrorString(err);
                TMS_LOG_WARN("pre_allocate: cudaMallocHost failed, using unpinned malloc."
                          << " tag=" << metadata.tag
                          << " size=" << (metadata.size / 1024 / 1024) << "MB"
                          << " err=" << err << " (" << (err_str ? err_str : "?") << ")");
                cudaGetLastError();  // clear sticky error
            } else {
                pinned_ok = true;
            }
        }

        if (!pinned_ok) {
            metadata.cpu_backup = std::malloc(metadata.size);
            if (metadata.cpu_backup == nullptr) {
                TMS_LOG_ERROR("FATAL: pre_allocate: malloc failed."
                          << " tag=" << metadata.tag
                          << " size=" << (metadata.size / 1024 / 1024) << "MB");
                exit(1);
            }
            metadata.cpu_backup_is_pinned = false;
            unpinned_count++;
        } else {
            metadata.cpu_backup_is_pinned = true;
            pinned_count++;
        }
        count++;
        total_bytes += metadata.size;
    }

    // Always print this summary (even at log level 0) since it is called
    // explicitly by the user and the output is valuable for debugging.
    std::cout << "[TMS INFO] pre_allocate_cpu_backup: filter_tag=" << (tag.empty() ? "(all)" : tag)
              << " newly_allocated=" << count
              << " total_bytes=" << (total_bytes / 1024 / 1024) << "MB"
              << " pinned=" << pinned_count
              << " unpinned_fallback=" << unpinned_count
              << " already_existed=" << skipped_count
              << std::endl;
}
