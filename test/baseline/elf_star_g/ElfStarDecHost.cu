//
//ElfStarDecHost.cu -
//

#include <vector>
#include <memory>
#include <cstring>
#include <cuda_runtime.h>
#include "BitStream/BitReader.cuh"
#include "Elf_Star_g_Kernel.cuh"

//CUDA （ ）
class CudaMemoryManagerDecoder {
private:
    std::vector<void*> allocated_ptrs;
    bool valid_context;
    
public:
    CudaMemoryManagerDecoder() : valid_context(true) {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            //printf(" CUDA : %s\n", cudaGetErrorString(err));
            valid_context = false;
        }
    }
    
    template<typename T>
    T* allocate(size_t count) {
        if (!valid_context) {
            //printf(" CUDA ， \n");
            return nullptr;
        }
        
        T* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, count * sizeof(T));
        if (err != cudaSuccess) {
            //printf(" cudaMalloc : %s\n", cudaGetErrorString(err));
            if (err == cudaErrorDeviceUninitialized || err == cudaErrorInvalidDevice) {
                valid_context = false;
                //printf(" CUDA ， \n");
            }
            return nullptr;
        }
        
        if (ptr != nullptr) {
            allocated_ptrs.push_back(ptr);
        }
        return ptr;
    }
    
    bool isValid() const {
        return valid_context;
    }
    
    void clear() {
        for (void* ptr : allocated_ptrs) {
            if (ptr) {
                cudaError_t err = cudaFree(ptr);
                if (err != cudaSuccess) {
                    //printf(" cudaFree : %s\n", cudaGetErrorString(err));
                }
            }
        }
        allocated_ptrs.clear();
    }
    
    ~CudaMemoryManagerDecoder() {
        clear();
    }
};

//CUDA （ ）
bool checkCudaDeviceStateForDecode() {
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    
    if (err != cudaSuccess) {
        //printf(" CUDA : %s\n", cudaGetErrorString(err));
        cudaGetLastError();
        
        err = cudaMemGetInfo(&free_mem, &total_mem);
        if (err != cudaSuccess) {
            //printf(" CUDA \n");
            return false;
        }
    }
    
    //printf(" GPU : %zu MB, %zu MB\n",
    //        free_mem/1024/1024, total_mem/1024/1024);
    
    return true;
}

//translated comment
int elf_star_parse_blocks_fixed(const uint8_t *compressed_data, ssize_t compressed_len,
                               int *out_num_blocks, size_t **out_block_sizes, 
                               size_t *out_total_elements) {
    if (!compressed_data || compressed_len < 4 || !out_num_blocks || !out_total_elements) {
        return -1;
    }

    //printf(" ： =%lld\n", (long long)compressed_len);

    std::vector<size_t> block_sizes;
    size_t total_elements = 0;
    size_t offset = 0;
    
    while (offset + 4 <= compressed_len) {
        uint32_t elements_in_block = *(reinterpret_cast<const uint32_t*>(compressed_data + offset));
        //printf(" %zu %u \n", offset, elements_in_block);
        
        if (elements_in_block == 0 || elements_in_block > CHUNK_SIZE + 100) {
            //printf(" ， \n");
            break;
        }
        
        total_elements += elements_in_block;
        
        size_t next_block_offset = 0;
        bool found_next = false;
        
        size_t min_block_size, max_block_size;
        
        if (elements_in_block == CHUNK_SIZE) {
            min_block_size = 400;
            max_block_size = 9000;
        } else if (elements_in_block < 100) {
            min_block_size = 12;
            max_block_size = elements_in_block * 12;
        } else {
            min_block_size = 4 + elements_in_block / 2;
            max_block_size = 4 + elements_in_block * 9;
        }
        
        if (offset + max_block_size > compressed_len) {
            max_block_size = compressed_len - offset;
        }
        
        for (size_t search_pos = offset + min_block_size; 
             search_pos + 4 <= compressed_len && search_pos < offset + max_block_size; 
             search_pos += 4) {
            
            uint32_t potential_elements = *(reinterpret_cast<const uint32_t*>(compressed_data + search_pos));
            
            bool is_likely_block = false;
            
            if (elements_in_block == CHUNK_SIZE && potential_elements == CHUNK_SIZE) {
                is_likely_block = true;
            } else if (elements_in_block == CHUNK_SIZE && potential_elements > 0 && potential_elements < CHUNK_SIZE) {
                is_likely_block = true;
            } else if (elements_in_block < CHUNK_SIZE && potential_elements > CHUNK_SIZE * 10) {
                is_likely_block = false;
            }
            
            if (is_likely_block) {
                next_block_offset = search_pos;
                found_next = true;
                //printf(" %zu ， =%u\n", search_pos, potential_elements);
                break;
            }
        }
        
        size_t current_block_size;
        if (found_next) {
            current_block_size = next_block_offset - offset;
        } else {
            current_block_size = compressed_len - offset;
            //printf(" ， \n");
        }
        
        //printf(" %zu \n", current_block_size);
        block_sizes.push_back(current_block_size);
        
        offset = found_next ? next_block_offset : compressed_len;
        
        if (block_sizes.size() > 200) {
            printf("块数量过多，停止解析\n");
            break;
        }
    }
    
    *out_num_blocks = block_sizes.size();
    *out_total_elements = total_elements;
    
    //printf(" ：%d ， %zu \n", *out_num_blocks, total_elements);
    // for (int i = 0; i < *out_num_blocks && i < 5; i++) {
    //printf(" %d ：%zu \n", i, block_sizes[i]);
    // }
    
    if (out_block_sizes && !block_sizes.empty()) {
        *out_block_sizes = (size_t*)malloc(block_sizes.size() * sizeof(size_t));
        if (*out_block_sizes) {
            memcpy(*out_block_sizes, block_sizes.data(), block_sizes.size() * sizeof(size_t));
        }
    }
    
    return 0;
}

//translated comment
ssize_t elf_star_decode_with_timing(const uint8_t *all_in_data,
                                   const size_t *in_offsets_bytes,
                                   const size_t *in_lengths_bytes,
                                   double *all_out_data,
                                   const size_t *out_offsets,
                                   int num_blocks,
                                   ElfStarTimingInfo *timing_info) {
    if (num_blocks <= 0) {
        return 0;
    }
    
    //translated comment
    if (timing_info) {
        timing_info->decompress_h2d_time = 0.0f;
        timing_info->decompress_kernel_time = 0.0f;
        timing_info->decompress_d2h_time = 0.0f;
        timing_info->total_decompress_time = 0.0f;
    }
    
    //CUDA
    if (!checkCudaDeviceStateForDecode()) {
        //printf(" CUDA ， \n");
        return -1;
    }
    
    const int64_t total_in_bytes = in_offsets_bytes[num_blocks];
    
    int64_t total_out_elements = 0;
    for (int i = 0; i < num_blocks; ++i) {
        if (in_offsets_bytes[i] + 4 <= total_in_bytes) {
            const uint8_t *chunk_start = all_in_data + in_offsets_bytes[i];
            uint32_t elements = *(reinterpret_cast<const uint32_t *>(chunk_start));
            if (elements > 0 && elements <= CHUNK_SIZE + 100) {
                total_out_elements += elements;
            } else {
                //printf(" %d : %u\n", i, elements);
                return -1;
            }
        } else {
            //printf(" %d \n", i);
            return -1;
        }
    }

    const int64_t total_out_bytes = total_out_elements * sizeof(double);
    
    //printf(" : %lld , %lld \n",
    //        (long long)total_in_bytes, (long long)total_out_bytes);

    //translated comment
    CudaMemoryManagerDecoder cuda_mem;
    if (!cuda_mem.isValid()) {
        //printf(" CUDA \n");
        return -1;
    }

    //CUDA
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //translated comment
    cudaEvent_t decompress_start, decompress_end;
    cudaEvent_t h2d_start, h2d_end;
    cudaEvent_t kernel_start, kernel_end;
    cudaEvent_t d2h_start, d2h_end;

    cudaEventCreate(&decompress_start);
    cudaEventCreate(&decompress_end);
    cudaEventCreate(&h2d_start);
    cudaEventCreate(&h2d_end);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_end);
    cudaEventCreate(&d2h_start);
    cudaEventCreate(&d2h_end);

    const int threads_per_block = 256;
    const int blocks_per_grid = (num_blocks + threads_per_block - 1) / threads_per_block;

    //translated comment
    uint8_t* d_in_data = cuda_mem.allocate<uint8_t>(total_in_bytes);
    if (!d_in_data) {
        //printf(" \n");
        //translated comment
        cudaEventDestroy(decompress_start);
        cudaEventDestroy(decompress_end);
        cudaEventDestroy(h2d_start);
        cudaEventDestroy(h2d_end);
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_end);
        cudaEventDestroy(d2h_start);
        cudaEventDestroy(d2h_end);
        cudaStreamDestroy(stream);
        return -1;
    }
    
    double* d_out_data = cuda_mem.allocate<double>(total_out_elements);
    if (!d_out_data) {
        //printf(" \n");
        //translated comment
        cudaEventDestroy(decompress_start);
        cudaEventDestroy(decompress_end);
        cudaEventDestroy(h2d_start);
        cudaEventDestroy(h2d_end);
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_end);
        cudaEventDestroy(d2h_start);
        cudaEventDestroy(d2h_end);
        cudaStreamDestroy(stream);
        return -1;
    }
    
    size_t* d_in_offsets = cuda_mem.allocate<size_t>(num_blocks + 1);
    if (!d_in_offsets) {
        //printf(" \n");
        //translated comment
        cudaEventDestroy(decompress_start);
        cudaEventDestroy(decompress_end);
        cudaEventDestroy(h2d_start);
        cudaEventDestroy(h2d_end);
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_end);
        cudaEventDestroy(d2h_start);
        cudaEventDestroy(d2h_end);
        cudaStreamDestroy(stream);
        return -1;
    }
    
    size_t* d_out_offsets = cuda_mem.allocate<size_t>(num_blocks);
    if (!d_out_offsets) {
        //printf(" \n");
        //translated comment
        cudaEventDestroy(decompress_start);
        cudaEventDestroy(decompress_end);
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
    cudaEventRecord(decompress_start, stream);
    
    //H2D: Host to Device
    cudaEventRecord(h2d_start, stream);
    
    cudaError_t cuda_err;
    cuda_err = cudaMemcpyAsync(d_in_data, all_in_data, total_in_bytes, cudaMemcpyHostToDevice, stream);
    if (cuda_err != cudaSuccess) {
        //printf(" : %s\n", cudaGetErrorString(cuda_err));
        //translated comment
        cudaEventDestroy(decompress_start);
        cudaEventDestroy(decompress_end);
        cudaEventDestroy(h2d_start);
        cudaEventDestroy(h2d_end);
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_end);
        cudaEventDestroy(d2h_start);
        cudaEventDestroy(d2h_end);
        cudaStreamDestroy(stream);
        return -1;
    }
    
    cuda_err = cudaMemcpyAsync(d_in_offsets, in_offsets_bytes, (num_blocks + 1) * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    if (cuda_err != cudaSuccess) {
        //printf(" : %s\n", cudaGetErrorString(cuda_err));
        //translated comment
        cudaEventDestroy(decompress_start);
        cudaEventDestroy(decompress_end);
        cudaEventDestroy(h2d_start);
        cudaEventDestroy(h2d_end);
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_end);
        cudaEventDestroy(d2h_start);
        cudaEventDestroy(d2h_end);
        cudaStreamDestroy(stream);
        return -1;
    }
    
    cuda_err = cudaMemcpyAsync(d_out_offsets, out_offsets, num_blocks * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    if (cuda_err != cudaSuccess) {
        //printf(" : %s\n", cudaGetErrorString(cuda_err));
        //translated comment
        cudaEventDestroy(decompress_start);
        cudaEventDestroy(decompress_end);
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
    
    //printf(" ， : %d\n", num_blocks);
    
    decompress_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        d_in_data, d_in_offsets, d_out_data, d_out_offsets, num_blocks);
    
    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        //printf(" : %s\n", cudaGetErrorString(cuda_err));
        //translated comment
        cudaEventDestroy(decompress_start);
        cudaEventDestroy(decompress_end);
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
    
    cuda_err = cudaMemcpyAsync(all_out_data, d_out_data, total_out_bytes, cudaMemcpyDeviceToHost, stream);
    if (cuda_err != cudaSuccess) {
        //printf(" : %s\n", cudaGetErrorString(cuda_err));
        //translated comment
        cudaEventDestroy(decompress_start);
        cudaEventDestroy(decompress_end);
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
    cudaEventRecord(decompress_end, stream);

    //translated comment
    cuda_err = cudaStreamSynchronize(stream);
    if (cuda_err != cudaSuccess) {
        //printf(" : %s\n", cudaGetErrorString(cuda_err));
        
        if (cuda_err == cudaErrorDeviceUninitialized || cuda_err == cudaErrorInvalidDevice) {
            //printf(" CUDA \n");
            cudaDeviceReset();
        }
        //translated comment
        cudaEventDestroy(decompress_start);
        cudaEventDestroy(decompress_end);
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
        cudaEventElapsedTime(&timing_info->decompress_h2d_time, h2d_start, h2d_end);
        cudaEventElapsedTime(&timing_info->decompress_kernel_time, kernel_start, kernel_end);
        cudaEventElapsedTime(&timing_info->decompress_d2h_time, d2h_start, d2h_end);
        cudaEventElapsedTime(&timing_info->total_decompress_time, decompress_start, decompress_end);
    }

    //printf(" \n");
    
    //translated comment
    if (!checkCudaDeviceStateForDecode()) {
        //printf(" ： CUDA \n");
    }
    
    //translated comment
    cudaEventDestroy(decompress_start);
    cudaEventDestroy(decompress_end);
    cudaEventDestroy(h2d_start);
    cudaEventDestroy(h2d_end);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_end);
    cudaEventDestroy(d2h_start);
    cudaEventDestroy(d2h_end);
    cudaStreamDestroy(stream);
    
    return total_out_bytes;
}

//translated comment
ssize_t elf_star_decode_simple_with_timing(const uint8_t *compressed_data, ssize_t compressed_len, 
                                          double **out, ssize_t *out_len,
                                          ElfStarTimingInfo *timing_info) {
    if (!compressed_data || compressed_len <= 0 || !out || !out_len) {
        return -1;
    }

    //translated comment
    int num_blocks = 0;
    size_t *block_sizes = nullptr;
    size_t total_elements = 0;
    
    if (elf_star_parse_blocks_fixed(compressed_data, compressed_len, &num_blocks, 
                                   &block_sizes, &total_elements) != 0) {
        //printf(" \n");
        return -1;
    }

    if (num_blocks <= 0 || total_elements <= 0) {
        //printf(" ： %d， %zu\n", num_blocks, total_elements);
        if (block_sizes) free(block_sizes);
        return -1;
    }

    //printf(" %d ， %zu \n", num_blocks, total_elements);

    //translated comment
    std::vector<size_t> in_offsets(num_blocks + 1);
    std::vector<size_t> out_offsets(num_blocks);
    
    in_offsets[0] = 0;
    out_offsets[0] = 0;
    
    size_t current_out_offset = 0;
    for (int i = 0; i < num_blocks; i++) {
        in_offsets[i + 1] = in_offsets[i] + block_sizes[i];
        out_offsets[i] = current_out_offset;
        
        //translated comment
        if (in_offsets[i] + 4 <= compressed_len) {
            uint32_t elements_in_block = *(reinterpret_cast<const uint32_t*>(compressed_data + in_offsets[i]));
            current_out_offset += elements_in_block;
        }
    }

    //translated comment
    *out = (double*)malloc(total_elements * sizeof(double));
    if (!*out) {
        //printf(" \n");
        if (block_sizes) free(block_sizes);
        return -1;
    }

    //translated comment
    ssize_t result = elf_star_decode_with_timing(compressed_data, in_offsets.data(), block_sizes,
                                               *out, out_offsets.data(), num_blocks,
                                               timing_info);

    if (result > 0) {
        *out_len = total_elements;
        //printf(" ：%zu \n", total_elements);
    } else {
        free(*out);
        *out = nullptr;
        *out_len = 0;
        //printf(" \n");
    }

    if (block_sizes) free(block_sizes);
    
    return result > 0 ? (ssize_t)(*out_len) : (ssize_t)-1;
}

//translated comment
ssize_t elf_star_decode(const uint8_t *all_in_data,
                        const size_t *in_offsets_bytes,
                        const size_t *in_lengths_bytes,
                        double *all_out_data,
                        const size_t *out_offsets,
                        int num_blocks) {
    return elf_star_decode_with_timing(all_in_data, in_offsets_bytes, in_lengths_bytes,
                                     all_out_data, out_offsets, num_blocks, nullptr);
}

ssize_t elf_star_decode_simple(const uint8_t *compressed_data, ssize_t compressed_len, 
                              double **out, ssize_t *out_len) {
    return elf_star_decode_simple_with_timing(compressed_data, compressed_len, out, out_len, nullptr);
}

void elf_star_free_encode_result(uint8_t *compressed_data, 
                                int64_t *compressed_lengths,
                                int64_t *compressed_offsets,
                                int64_t *decompressed_offsets) {
    if (compressed_data) free(compressed_data);
    if (compressed_lengths) free(compressed_lengths);
    if (compressed_offsets) free(compressed_offsets);
    if (decompressed_offsets) free(decompressed_offsets);
}