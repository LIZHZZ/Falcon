//
//ElfStarComHost.cu -
//

#include <vector>
#include <memory>
#include <cmath>
#include <cuda_runtime.h>
#include "Elf_Star_g_Kernel_32.cuh"

//CUDA （ ）
class SafeCudaMemoryManager_32 {
private:
    std::vector<void*> allocated_ptrs;
    bool cuda_available;
    
public:
    SafeCudaMemoryManager_32() : cuda_available(true) {
        cudaGetLastError();
        
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            cuda_available = false;
            //printf("CUDA : %s\n", cudaGetErrorString(err));
        } else {
            err = cudaSetDevice(0);
            if (err != cudaSuccess) {
                cuda_available = false;
                //printf(" CUDA : %s\n", cudaGetErrorString(err));
            }
        }
    }
    
    template<typename T>
    T* allocate(size_t count) {
        if (!cuda_available) {
            //printf("CUDA ， \n");
            return nullptr;
        }
        
        T* ptr = nullptr;
        size_t bytes = count * sizeof(T);
        
        size_t free_mem, total_mem;
        cudaError_t mem_err = cudaMemGetInfo(&free_mem, &total_mem);
        if (mem_err != cudaSuccess) {
            //printf(" GPU : %s\n", cudaGetErrorString(mem_err));
            cuda_available = false;
            return nullptr;
        }
        
        if (bytes > free_mem) {
            //printf("GPU : %zu MB, %zu MB\n",
            //        bytes/1024/1024, free_mem/1024/1024);
            return nullptr;
        }
        
        cudaError_t err = cudaMalloc(&ptr, bytes);
        if (err == cudaSuccess && ptr != nullptr) {
            allocated_ptrs.push_back(ptr);
            return ptr;
        } else {
            //printf("cudaMalloc (%zu ): %s\n", bytes, cudaGetErrorString(err));
            if (err == cudaErrorMemoryAllocation) {
                cuda_available = false;
            }
            return nullptr;
        }
    }
    
    bool isAvailable() const {
        return cuda_available;
    }
    
    void clear() {
        for (void* ptr : allocated_ptrs) {
            if (ptr) {
                cudaError_t err = cudaFree(ptr);
                if (err != cudaSuccess) {
                    //printf("cudaFree : %s\n", cudaGetErrorString(err));
                }
            }
        }
        allocated_ptrs.clear();
    }
    
    ~SafeCudaMemoryManager_32() {
        clear();
    }
};

//translated comment
ssize_t elf_star_encode_with_timing_32(float *in, ssize_t len, uint8_t **out, 
                                   int64_t **out_compressed_lengths,
                                   int64_t **out_compressed_offsets, 
                                   int64_t **out_decompressed_offsets, 
                                   int *out_num_blocks,
                                   ElfStarTimingInfo_32 *timing_info) {
    if (!in || len <= 0 || !out || !out_num_blocks) {
        printf("压缩参数无效\n");
        return -1;
    }

    //translated comment
    if (timing_info) {
        memset(timing_info, 0, sizeof(ElfStarTimingInfo_32));
    }

    //translated comment
    size_t valid_count = 0;
    for (ssize_t i = 0; i < len; i++) {
        if (!isnan(in[i]) && !isinf(in[i])) {
            valid_count++;
        }
    }
    
    if (valid_count == 0) {
        //printf(" \n");
        return -1;
    }
    
    //printf(" : %lld , %zu \n", (long long)len, valid_count);

    const size_t max_chunk_len_elems = CHUNK_SIZE;
    const int num_chunks = (len + max_chunk_len_elems - 1) / max_chunk_len_elems;
    *out_num_blocks = num_chunks;

    //translated comment
    std::vector<size_t> h_in_offsets(num_chunks + 1);
    std::vector<size_t> h_out_offsets(num_chunks + 1);
    std::vector<size_t> h_compressed_sizes(num_chunks);
    
    for (int i = 0; i < num_chunks; i++) {
        h_in_offsets[i] = i * max_chunk_len_elems;
    }
    h_in_offsets[num_chunks] = len;
    
    const size_t bytes_per_element = 12;
    const size_t estimated_out_bytes = len * bytes_per_element;
    const size_t out_chunk_size = (estimated_out_bytes + num_chunks - 1) / num_chunks;
    const size_t aligned_out_chunk_size = ((out_chunk_size + 3) / 4) * 4;
    
    for (int i = 0; i <= num_chunks; i++) {
        h_out_offsets[i] = i * aligned_out_chunk_size;
    }

    SafeCudaMemoryManager_32 cuda_mem;
    if (!cuda_mem.isAvailable()) {
        printf("CUDA环境不可用\n");
        return -1;
    }

    //CUDA
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //translated comment
    cudaEvent_t compress_start, compress_end;
    cudaEvent_t h2d_start, h2d_end;
    cudaEvent_t kernel_start, kernel_end;
    cudaEvent_t d2h_start, d2h_end;

    cudaEventCreate(&compress_start);
    cudaEventCreate(&compress_end);
    cudaEventCreate(&h2d_start);
    cudaEventCreate(&h2d_end);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_end);
    cudaEventCreate(&d2h_start);
    cudaEventCreate(&d2h_end);

    printf("分配GPU内存...\n");
    
    //translated comment
    float* d_in_data = cuda_mem.allocate<float>(len);
    if (!d_in_data) return -1;

    const size_t total_out_bytes = num_chunks * aligned_out_chunk_size;
    uint8_t* d_out_data = cuda_mem.allocate<uint8_t>(total_out_bytes);
    if (!d_out_data) return -1;

    size_t* d_in_offsets = cuda_mem.allocate<size_t>(num_chunks + 1);
    if (!d_in_offsets) return -1;
    
    size_t* d_out_offsets = cuda_mem.allocate<size_t>(num_chunks + 1);
    if (!d_out_offsets) return -1;
    
    size_t* d_compressed_sizes = cuda_mem.allocate<size_t>(num_chunks);
    if (!d_compressed_sizes) return -1;
    
    const size_t temp_storage_per_chunk = max_chunk_len_elems * (sizeof(int) + sizeof(uint32_t));
    const size_t total_temp_storage = num_chunks * temp_storage_per_chunk;
    
    uint8_t* d_temp_storage = cuda_mem.allocate<uint8_t>(total_temp_storage);
    if (!d_temp_storage) return -1;

    //printf("GPU \n");

    //translated comment
    cudaEventRecord(compress_start, stream);
    
    //H2D: Host to Device
    cudaEventRecord(h2d_start, stream);
    
    cudaError_t cuda_err;
    cuda_err = cudaMemcpyAsync(d_in_data, in, len * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (cuda_err != cudaSuccess) {
        //printf(" : %s\n", cudaGetErrorString(cuda_err));
        //translated comment
        cudaEventDestroy(compress_start);
        cudaEventDestroy(compress_end);
        cudaEventDestroy(h2d_start);
        cudaEventDestroy(h2d_end);
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_end);
        cudaEventDestroy(d2h_start);
        cudaEventDestroy(d2h_end);
        cudaStreamDestroy(stream);
        return -1;
    }
    
    cuda_err = cudaMemcpyAsync(d_in_offsets, h_in_offsets.data(), 
                              (num_chunks + 1) * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    if (cuda_err != cudaSuccess) {
        //printf(" : %s\n", cudaGetErrorString(cuda_err));
        //translated comment
        cudaEventDestroy(compress_start);
        cudaEventDestroy(compress_end);
        cudaEventDestroy(h2d_start);
        cudaEventDestroy(h2d_end);
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_end);
        cudaEventDestroy(d2h_start);
        cudaEventDestroy(d2h_end);
        cudaStreamDestroy(stream);
        return -1;
    }
    
    cuda_err = cudaMemcpyAsync(d_out_offsets, h_out_offsets.data(), 
                              (num_chunks + 1) * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    if (cuda_err != cudaSuccess) {
        //printf(" : %s\n", cudaGetErrorString(cuda_err));
        //translated comment
        cudaEventDestroy(compress_start);
        cudaEventDestroy(compress_end);
        cudaEventDestroy(h2d_start);
        cudaEventDestroy(h2d_end);
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_end);
        cudaEventDestroy(d2h_start);
        cudaEventDestroy(d2h_end);
        cudaStreamDestroy(stream);
        return -1;
    }
    
    cudaEventRecord(h2d_end, stream);
    
    //KERNEL:
    cudaEventRecord(kernel_start, stream);
    
    const int threads_per_block = 256;
    const int blocks_per_grid = (num_chunks + threads_per_block - 1) / threads_per_block;
    
    //printf(" : %d , %d \n", blocks_per_grid, threads_per_block);
    
    compress_kernel_32<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        d_in_data, d_in_offsets, d_out_data, d_out_offsets,
        d_compressed_sizes, d_temp_storage, max_chunk_len_elems, num_chunks
    );

    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        //printf(" : %s\n", cudaGetErrorString(cuda_err));
        //translated comment
        cudaEventDestroy(compress_start);
        cudaEventDestroy(compress_end);
        cudaEventDestroy(h2d_start);
        cudaEventDestroy(h2d_end);
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_end);
        cudaEventDestroy(d2h_start);
        cudaEventDestroy(d2h_end);
        cudaStreamDestroy(stream);
        return -1;
    }

    cudaEventRecord(kernel_end, stream);

    //D2H: Device to Host
    cudaEventRecord(d2h_start, stream);
    
    cuda_err = cudaMemcpyAsync(h_compressed_sizes.data(), d_compressed_sizes, 
                              num_chunks * sizeof(size_t), cudaMemcpyDeviceToHost, stream);
    if (cuda_err != cudaSuccess) {
        //printf(" : %s\n", cudaGetErrorString(cuda_err));
        //translated comment
        cudaEventDestroy(compress_start);
        cudaEventDestroy(compress_end);
        cudaEventDestroy(h2d_start);
        cudaEventDestroy(h2d_end);
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_end);
        cudaEventDestroy(d2h_start);
        cudaEventDestroy(d2h_end);
        cudaStreamDestroy(stream);
        return -1;
    }
    
    cudaEventRecord(d2h_end, stream);
    cudaEventRecord(compress_end, stream);

    //translated comment
    cuda_err = cudaStreamSynchronize(stream);
    if (cuda_err != cudaSuccess) {
        //printf(" : %s\n", cudaGetErrorString(cuda_err));
        //translated comment
        cudaEventDestroy(compress_start);
        cudaEventDestroy(compress_end);
        cudaEventDestroy(h2d_start);
        cudaEventDestroy(h2d_end);
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_end);
        cudaEventDestroy(d2h_start);
        cudaEventDestroy(d2h_end);
        cudaStreamDestroy(stream);
        return -1;
    }

    //translated comment
    if (timing_info) {
        cudaEventElapsedTime(&timing_info->compress_h2d_time, h2d_start, h2d_end);
        cudaEventElapsedTime(&timing_info->compress_kernel_time, kernel_start, kernel_end);
        cudaEventElapsedTime(&timing_info->compress_d2h_time, d2h_start, d2h_end);
        cudaEventElapsedTime(&timing_info->total_compress_time, compress_start, compress_end);
    }

    //printf(" \n");

    //translated comment
    size_t total_compressed_bytes = 0;
    int successful_chunks = 0;
    
    for (int i = 0; i < num_chunks; i++) {
        if (h_compressed_sizes[i] > 0) {
            total_compressed_bytes += h_compressed_sizes[i];
            successful_chunks++;
        } else {
            //printf(" : %d \n", i);
        }
    }
    
    if (successful_chunks == 0) {
        //printf(" \n");
        //translated comment
        cudaEventDestroy(compress_start);
        cudaEventDestroy(compress_end);
        cudaEventDestroy(h2d_start);
        cudaEventDestroy(h2d_end);
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_end);
        cudaEventDestroy(d2h_start);
        cudaEventDestroy(d2h_end);
        cudaStreamDestroy(stream);
        return -1;
    }
    
    //printf(" %d/%d ， %zu \n",
    //        successful_chunks, num_chunks, total_compressed_bytes);

    //translated comment
    std::vector<size_t> h_actual_out_offsets(num_chunks + 1);
    h_actual_out_offsets[0] = 0;
    for (int i = 0; i < num_chunks; i++) {
        h_actual_out_offsets[i + 1] = h_actual_out_offsets[i] + h_compressed_sizes[i];
    }

    //translated comment
    *out = (uint8_t*)malloc(total_compressed_bytes);
    if (!*out) {
        //printf(" \n");
        //translated comment
        cudaEventDestroy(compress_start);
        cudaEventDestroy(compress_end);
        cudaEventDestroy(h2d_start);
        cudaEventDestroy(h2d_end);
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_end);
        cudaEventDestroy(d2h_start);
        cudaEventDestroy(d2h_end);
        cudaStreamDestroy(stream);
        return -1;
    }

    //translated comment
    for (int i = 0; i < num_chunks; i++) {
        if (h_compressed_sizes[i] > 0) {
            cuda_err = cudaMemcpy(*out + h_actual_out_offsets[i], 
                                 d_out_data + h_out_offsets[i], 
                                 h_compressed_sizes[i], cudaMemcpyDeviceToHost);
            if (cuda_err != cudaSuccess) {
                //printf(" %d : %s\n", i, cudaGetErrorString(cuda_err));
                free(*out);
                *out = nullptr;
                //translated comment
                cudaEventDestroy(compress_start);
                cudaEventDestroy(compress_end);
                cudaEventDestroy(h2d_start);
                cudaEventDestroy(h2d_end);
                cudaEventDestroy(kernel_start);
                cudaEventDestroy(kernel_end);
                cudaEventDestroy(d2h_start);
                cudaEventDestroy(d2h_end);
                cudaStreamDestroy(stream);
                return -1;
            }
        }
    }

    //translated comment
    if (out_compressed_lengths) {
        *out_compressed_lengths = (int64_t*)malloc(num_chunks * sizeof(int64_t));
        if (*out_compressed_lengths) {
            for (int i = 0; i < num_chunks; i++) {
                (*out_compressed_lengths)[i] = h_compressed_sizes[i];
            }
        }
    }

    if (out_compressed_offsets) {
        *out_compressed_offsets = (int64_t*)malloc((num_chunks + 1) * sizeof(int64_t));
        if (*out_compressed_offsets) {
            for (int i = 0; i <= num_chunks; i++) {
                (*out_compressed_offsets)[i] = h_actual_out_offsets[i];
            }
        }
    }

    if (out_decompressed_offsets) {
        *out_decompressed_offsets = (int64_t*)malloc(num_chunks * sizeof(int64_t));
        if (*out_decompressed_offsets) {
            for (int i = 0; i < num_chunks; i++) {
                (*out_decompressed_offsets)[i] = i * max_chunk_len_elems;
            }
        }
    }

    double compression_ratio = (double)(len * sizeof(float)) / total_compressed_bytes;
    //printf(" : %.2f:1 ， %.1f%%\n",
    //        compression_ratio, (1.0 - (double)total_compressed_bytes / (len * sizeof(double))) * 100.0);

    //translated comment
    cudaEventDestroy(compress_start);
    cudaEventDestroy(compress_end);
    cudaEventDestroy(h2d_start);
    cudaEventDestroy(h2d_end);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_end);
    cudaEventDestroy(d2h_start);
    cudaEventDestroy(d2h_end);
    cudaStreamDestroy(stream);

    return (ssize_t)total_compressed_bytes;
}

//translated comment
ssize_t elf_star_encode_simple_with_timing_32(const float *in, ssize_t len, 
                                          uint8_t **out, ssize_t *out_len,
                                          ElfStarTimingInfo_32 *timing_info) {
    if (!in || len <= 0 || !out || !out_len) {
        return -1;
    }

    uint8_t *compressed_data = nullptr;
    int64_t *compressed_lengths = nullptr;
    int64_t *compressed_offsets = nullptr; 
    int64_t *decompressed_offsets = nullptr;
    int num_blocks = 0;

    ssize_t result = elf_star_encode_with_timing_32((float*)in, len, &compressed_data,
                                               &compressed_lengths, &compressed_offsets,
                                               &decompressed_offsets, &num_blocks,
                                               timing_info);

    if (result > 0) {
        *out = compressed_data;
        *out_len = result;
        
        //printf(" : %lld -> %lld \n",
        //        (long long)(len * sizeof(double)), (long long)result);
    } else {
        *out = nullptr;
        *out_len = 0;
        //printf(" \n");
    }

    //translated comment
    if (compressed_lengths) free(compressed_lengths);
    if (compressed_offsets) free(compressed_offsets);
    if (decompressed_offsets) free(decompressed_offsets);

    return result;
}

//translated comment
ssize_t elf_star_encode_32(float *in, ssize_t len, uint8_t **out, 
                        int64_t **out_compressed_lengths,
                        int64_t **out_compressed_offsets, 
                        int64_t **out_decompressed_offsets, 
                        int *out_num_blocks) {
    return elf_star_encode_with_timing_32(in, len, out, out_compressed_lengths,
                                     out_compressed_offsets, out_decompressed_offsets,
                                     out_num_blocks, nullptr);
}

ssize_t elf_star_encode_simple(const float *in, ssize_t len, uint8_t **out, ssize_t *out_len) {
    return elf_star_encode_simple_with_timing_32(in, len, out, out_len, nullptr);
}