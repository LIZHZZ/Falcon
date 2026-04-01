//
//Elf_Star_g_Kernel.cuh -
//

#ifndef ELF_STAR_G_KERNEL_CUH
#define ELF_STAR_G_KERNEL_CUH
#include <cuda/std/cstdint>
#include <defs.cuh>
#include <cstdint>
#include <cstdio>
#define CHUNK_SIZE 1024

#define LOG_2_10 3.32192809489
#define MAX_CHUNK_BYTES 8192

//translated comment
struct ElfStarTimingInfo {
    float compress_h2d_time;      //H2D (ms)
    float compress_kernel_time;   //translated comment
    float compress_d2h_time;      //D2H (ms)
    float decompress_h2d_time;    //H2D (ms)
    float decompress_kernel_time; //translated comment
    float decompress_d2h_time;    //D2H (ms)
    float total_compress_time;    //translated comment
    float total_decompress_time;  //translated comment
    
    void print() const {
        printf("=== ELF Star 性能统计 ===\n");
        printf("压缩阶段:\n");
        printf("  H2D 时间: %.3f ms\n", compress_h2d_time);
        printf("  核函数时间: %.3f ms\n", compress_kernel_time);
        printf("  D2H 时间: %.3f ms\n", compress_d2h_time);
        printf("  总计: %.3f ms\n", total_compress_time);
        printf("解压阶段:\n");
        printf("  H2D 时间: %.3f ms\n", decompress_h2d_time);
        printf("  核函数时间: %.3f ms\n", decompress_kernel_time);
        printf("  D2H 时间: %.3f ms\n", decompress_d2h_time);
        printf("  总计: %.3f ms\n", total_decompress_time);
        printf("性能占比分析:\n");
        float total_time = total_compress_time + total_decompress_time;
        if (total_time > 0) {
            printf("  压缩核函数占比: %.1f%%\n", (compress_kernel_time / total_time) * 100.0);
            printf("  解压核函数占比: %.1f%%\n", (decompress_kernel_time / total_time) * 100.0);
            printf("  数据传输占比: %.1f%%\n", 
                   ((compress_h2d_time + compress_d2h_time + decompress_h2d_time + decompress_d2h_time) / total_time) * 100.0);
        }
        printf("========================\n");
    }
};

//translated comment

/**
 *@brief GPU
 */
__global__ void decompress_kernel(const uint8_t *d_in_data,
                                  const size_t *d_in_offsets,
                                  double *d_out_data,
                                  const size_t *d_out_offsets,
                                  int num_chunks);

/**
 *@brief GPU
 */
__global__ void compress_kernel(const double *d_in_data,
                                const size_t *d_in_offsets,
                                uint8_t *d_out_data,
                                const size_t *d_out_offsets,
                                size_t *d_compressed_sizes_bytes,
                                uint8_t *d_temp_storage,
                                size_t max_chunk_len_elems,
                                int num_chunks);

//translated comment

/**
 *@brief
 */
ssize_t elf_star_encode_with_timing(double *in, ssize_t len, uint8_t **out, 
                                   int64_t **out_compressed_lengths,
                                   int64_t **out_compressed_offsets, 
                                   int64_t **out_decompressed_offsets, 
                                   int *out_num_blocks,
                                   ElfStarTimingInfo *timing_info);

/**
 *@brief
 */
ssize_t elf_star_encode_simple_with_timing(const double *in, ssize_t len, 
                                          uint8_t **out, ssize_t *out_len,
                                          ElfStarTimingInfo *timing_info);

/**
 *@brief
 */
ssize_t elf_star_decode_with_timing(const uint8_t *all_in_data,
                                   const size_t *in_offsets_bytes,
                                   const size_t *in_lengths_bytes,
                                   double *all_out_data,
                                   const size_t *out_offsets,
                                   int num_blocks,
                                   ElfStarTimingInfo *timing_info);

/**
 *@brief
 */
ssize_t elf_star_decode_simple_with_timing(const uint8_t *compressed_data, ssize_t compressed_len, 
                                          double **out, ssize_t *out_len,
                                          ElfStarTimingInfo *timing_info);

//translated comment

/**
 *@brief
 */
ssize_t elf_star_encode(double *in, ssize_t len, uint8_t **out, 
                        int64_t **out_compressed_lengths,
                        int64_t **out_compressed_offsets, 
                        int64_t **out_decompressed_offsets, 
                        int *out_num_blocks);

/**
 *@brief
 */
ssize_t elf_star_encode_simple(const double *in, ssize_t len, uint8_t **out, ssize_t *out_len);

/**
 *@brief
 */
ssize_t elf_star_decode(const uint8_t *all_in_data,
                        const size_t *in_offsets_bytes,
                        const size_t *in_lengths_bytes,
                        double *all_out_data,
                        const size_t *out_offsets,
                        int num_blocks);

/**
 *@brief
 */
ssize_t elf_star_decode_simple(const uint8_t *compressed_data, ssize_t compressed_len, 
                              double **out, ssize_t *out_len);

//translated comment

/**
 *@brief
 */
int elf_star_parse_blocks(const uint8_t *compressed_data, ssize_t compressed_len,
                         int *out_num_blocks, size_t **out_block_sizes, 
                         size_t *out_total_elements);

/**
 *@brief elf_star_encode
 */
void elf_star_free_encode_result(uint8_t *compressed_data, 
                                int64_t *compressed_lengths,
                                int64_t *compressed_offsets,
                                int64_t *decompressed_offsets);

#endif //ELF_STAR_G_KERNEL_CUH