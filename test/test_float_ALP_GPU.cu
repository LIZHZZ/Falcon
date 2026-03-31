// #include <gtest/gtest.h>
// #include <fstream>
// #include <vector>
// #include <array>
// #include <chrono>
// #include <iostream>
// #include <cstring>
// #include <cmath>
// #include <algorithm>
// #include <cuda_runtime.h>
// #include <filesystem>
// #include <iomanip>
// #include <numeric>
// #include <cassert>

// // ALP-G 头文件
// #include "alp/alp-bindings.cuh"
// #include "flsgpu/flsgpu-api.cuh"
// #include "flsgpu/structs.cuh"
// #include "data/dataset_utils.hpp"
// #include "generated-bindings/kernel-bindings.cuh"
// #include "engine/enums.cuh"
// #include "engine/data.cuh"
// #include "engine/verification.cuh"

// namespace fs = std::filesystem;

// // 函数声明(保持与基础版文件一致命名,内部改为扩展格式)
// CompressionInfo comp_ALP_G(std::vector<float> oriData);
// CompressionInfo test_compression(const std::string &file_path);
// CompressionInfo test_beta_compression(const std::string &file_path, int beta);

// template <typename T>
// bool can_use_extended(const flsgpu::host::ALPColumn<T> &column)
// {
//     constexpr size_t N_LANES = utils::get_n_lanes<T>();
//     constexpr uint16_t MAX_LANE_COUNT = (1u << 6) - 1; // offsets_counts count field is 6 bits

//     const size_t n_vecs = column.ffor.bp.get_n_vecs();
//     for (size_t vec = 0; vec < n_vecs; ++vec)
//     {
//         const uint32_t exc_count = column.counts[vec];
//         if (exc_count == 0)
//         {
//             continue;
//         }

//         const size_t exc_offset = column.exceptions_offsets[vec];
//         std::array<uint16_t, N_LANES> lane_counts{};
//         for (uint32_t i = 0; i < exc_count; ++i)
//         {
//             const uint16_t pos = column.positions[exc_offset + i];
//             const uint16_t lane = static_cast<uint16_t>(pos % N_LANES);
//             ++lane_counts[lane];
//         }

//         for (uint16_t lane_count : lane_counts)
//         {
//             if (lane_count > MAX_LANE_COUNT)
//             {
//                 return false;
//             }
//         }
//     }

//     return true;
// }

// // ==================== 扩展版压缩 + GPU 解压 ====================
// CompressionInfo comp_ALP_G(std::vector<float> oriData)
// {
//     const size_t original_num_elements = oriData.size();
//     const size_t original_size = original_num_elements * sizeof(float);
//     if (original_num_elements == 0)
//     {
//         std::cerr << "❌ 输入数据为空" << std::endl;
//         return CompressionInfo{};
//     }

//     // 与基础版统一的参数
//     constexpr size_t VECTOR_SIZE = 1024;
//     constexpr unsigned UNPACK_N_VECTORS = 1; // 安全配置
//     constexpr size_t N_LANES_FLOAT = 32;
//     constexpr size_t THREADS_PER_WARP = 32;
//     constexpr size_t N_WARPS_PER_BLOCK = 2;
//     constexpr size_t N_THREADS_PER_BLOCK = N_WARPS_PER_BLOCK * THREADS_PER_WARP;
//     constexpr size_t N_CONCURRENT_VECTORS_PER_BLOCK = N_THREADS_PER_BLOCK / N_LANES_FLOAT; // 64/32=2
//     constexpr size_t VECTORS_PER_BLOCK = UNPACK_N_VECTORS * N_CONCURRENT_VECTORS_PER_BLOCK; // 2

//     size_t num_elements = original_num_elements;
//     std::vector<float> paddedData;
//     const float *data_ptr = oriData.data();

//     // 1.数据预处理
//     // 填充到线程块向量组的倍数
//     size_t n_vecs = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
//     size_t n_vecs_padded = ((n_vecs + VECTORS_PER_BLOCK - 1) / VECTORS_PER_BLOCK) * VECTORS_PER_BLOCK;
//     size_t num_elements_padded = n_vecs_padded * VECTOR_SIZE;
//     if (num_elements_padded != original_num_elements)
//     {
//         size_t pad_needed = num_elements_padded - original_num_elements;
//         num_elements = num_elements_padded;
//         paddedData.reserve(num_elements);
//         paddedData.insert(paddedData.end(), oriData.begin(), oriData.end());
//         paddedData.insert(paddedData.end(), pad_needed, oriData.back());
//         data_ptr = paddedData.data();
//     }
//     // 2.压缩阶段
//     auto encode_start = std::chrono::high_resolution_clock::now();
//     flsgpu::host::ALPColumn<float> host_base_column;
//     try
//     {
//         host_base_column = alp::encode<float>(data_ptr, num_elements, false);
//     }
//     catch (const std::exception &e)
//     {
//         std::cerr << "❌ 基础压缩失败: " << e.what() << std::endl;
//         return CompressionInfo{};
//     }
//     // 计时
//     auto encode_end = std::chrono::high_resolution_clock::now();
//     const double compression_time_ms = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();
//     constexpr double compression_kernel_time_ms = 0.0;
    
//     // 判断使用扩展版 or 基础版
//     const bool use_extended = can_use_extended(host_base_column);
//     const size_t compressed_size = use_extended ? host_base_column.compressed_size_bytes_alp_extended
//                                                 : host_base_column.compressed_size_bytes_alp;
//     const double compression_ratio = static_cast<double>(compressed_size) / original_size;
//     std::cout << "✓ 压缩完成(" << (use_extended ? "扩展" : "基础") << "): "
//               << compressed_size << " bytes, 比率=" << compression_ratio << "x" << std::endl;

//     // 创建 CUDA 事件用于计时
//     cudaEvent_t kernel_start{};
//     cudaEvent_t kernel_stop{};
//     cudaEventCreate(&kernel_start);
//     cudaEventCreate(&kernel_stop);
//     auto decomp_start = std::chrono::high_resolution_clock::now();
//     // 解压过程
//     if (use_extended)
//     {

//         flsgpu::host::ALPExtendedColumn<float> host_extended_column = host_base_column.create_extended_column();
//         flsgpu::device::ALPExtendedColumn<float> device_extended_column;
//         // decomp_start = std::chrono::high_resolution_clock::now();
//         // 数据转移
//         try
//         {
//             device_extended_column = host_extended_column.copy_to_device();
//             cudaDeviceSynchronize();
//         }
//         catch (const std::exception &e)
//         {
//             std::cerr << "❌ 扩展列复制到 GPU 失败: " << e.what() << std::endl;
//             flsgpu::host::free_column(host_extended_column);
//             flsgpu::host::free_column(host_base_column);
//             return CompressionInfo{};
//         }
        
//         float kernel_elapsed_ms = 0.0f;
//         float *host_decompressed_data = nullptr;
//         // 解压缩
//         try
//         {
//             cudaEventRecord(kernel_start);
//             host_decompressed_data = bindings::decompress_column<float, flsgpu::device::ALPExtendedColumn<float>>(
//                 device_extended_column,
//                 UNPACK_N_VECTORS,
//                 1,
//                 enums::Unpacker::StatefulBranchless,
//                 enums::Patcher::PrefetchAll,
//                 1);
//             cudaDeviceSynchronize();
//             cudaEventRecord(kernel_stop);
//             cudaEventSynchronize(kernel_stop);
//             cudaEventElapsedTime(&kernel_elapsed_ms, kernel_start, kernel_stop);
//             if (!host_decompressed_data)
//             {
//                 throw std::runtime_error("扩展GPU解压返回 nullptr");
//             }
//         }
//         catch (const std::exception &e)
//         {
//             std::cerr << "❌ 扩展GPU解压失败: " << e.what() << std::endl;
//             if (host_decompressed_data)
//                 delete[] host_decompressed_data;
//             flsgpu::host::free_column(device_extended_column);
//             flsgpu::host::free_column(host_extended_column);
//             flsgpu::host::free_column(host_base_column);
//             cudaEventDestroy(kernel_start);
//             cudaEventDestroy(kernel_stop);
//             return CompressionInfo{};
//         }
//         auto decomp_end = std::chrono::high_resolution_clock::now();

//         // 时间统计
//         const double decompression_time_ms = std::chrono::duration<double, std::milli>(decomp_end - decomp_start).count();
//         const double decompression_kernel_time_ms = static_cast<double>(kernel_elapsed_ms);
        
//         // 验证解压数据
//         const uint8_t *padded_bytes = reinterpret_cast<const uint8_t *>(data_ptr);
//         const uint8_t *decomp_bytes = reinterpret_cast<const uint8_t *>(host_decompressed_data);
//         const size_t decomp_size = device_extended_column.n_values * sizeof(float);
//         if (memcmp(padded_bytes, decomp_bytes, decomp_size) != 0)
//         {
//             std::cout << "❌ 扩展格式数据验证失败" << std::endl;
//             const float *p = data_ptr;
//             const float *d = host_decompressed_data;
//             int shown = 0;
//             for (size_t i = 0; i < device_extended_column.n_values && shown < 10; ++i)
//             {
//                 if (std::abs(p[i] - d[i]) > 1e-10)
//                 {
//                     std::cout << "  不匹配[" << i << "]: " << p[i] << " vs " << d[i] << std::endl;
//                     ++shown;
//                 }
//             }
//         }
//         else
//         {
//             std::cout << "✓ 扩展格式数据验证成功" << std::endl;
//         }

//         // 结果返回
//         const double comp_tp = compression_time_ms > 0.0 ? (original_size / (1024.0 * 1024.0 * 1024.0)) / (compression_time_ms / 1000.0) : 0.0;
//         const double decomp_tp = decompression_time_ms > 0.0 ? (original_size / (1024.0 * 1024.0 * 1024.0)) / (decompression_time_ms / 1000.0) : 0.0;

//         CompressionInfo result{
//             original_size / (1024.0 * 1024.0),
//             compressed_size / (1024.0 * 1024.0),
//             compression_ratio,
//             compression_kernel_time_ms,
//             compression_time_ms,
//             comp_tp,
//             decompression_kernel_time_ms,
//             decompression_time_ms,
//             decomp_tp};

//         delete[] host_decompressed_data;
//         flsgpu::host::free_column(device_extended_column);
//         flsgpu::host::free_column(host_extended_column);
//         flsgpu::host::free_column(host_base_column);
//         cudaEventDestroy(kernel_start);
//         cudaEventDestroy(kernel_stop);
//         cudaDeviceSynchronize();
//         return result;
//     }
//     else{
//         // 数据传输
//         flsgpu::device::ALPColumn<float> device_base_column;
//         try
//         {
//             device_base_column = host_base_column.copy_to_device();
//             cudaDeviceSynchronize();
//         }
//         catch (const std::exception &e)
//         {
//             std::cerr << "❌ 基础列复制到 GPU 失败: " << e.what() << std::endl;
//             flsgpu::host::free_column(host_base_column);
//             return CompressionInfo{};
//         }

//         float kernel_elapsed_ms = 0.0f;
//         float *host_decompressed_data = nullptr;
//         // 基础版本解压缩
//         try
//         {
//             cudaEventRecord(kernel_start);
//             host_decompressed_data = bindings::decompress_column<float, flsgpu::device::ALPColumn<float>>(
//                 device_base_column,
//                 UNPACK_N_VECTORS,
//                 1,
//                 enums::Unpacker::StatefulBranchless,
//                 enums::Patcher::Stateful,
//                 1);
//             cudaEventRecord(kernel_stop);
//             cudaEventSynchronize(kernel_stop);
//             cudaEventElapsedTime(&kernel_elapsed_ms, kernel_start, kernel_stop);
//             cudaDeviceSynchronize();
//             if (!host_decompressed_data)
//             {
//                 throw std::runtime_error("基础GPU解压返回 nullptr");
//             }
//         }
//         catch (const std::exception &e)
//         {
//             std::cerr << "❌ 基础GPU解压失败: " << e.what() << std::endl;
//             if (host_decompressed_data)
//                 delete[] host_decompressed_data;
//             flsgpu::host::free_column(device_base_column);
//             flsgpu::host::free_column(host_base_column);
//             cudaEventDestroy(kernel_start);
//             cudaEventDestroy(kernel_stop);
//             return CompressionInfo{};
//         }
//         // 时间统计
//         auto decomp_end = std::chrono::high_resolution_clock::now();
//         const double decompression_time_ms = std::chrono::duration<double, std::milli>(decomp_end - decomp_start).count();
//         const double decompression_kernel_time_ms = static_cast<double>(kernel_elapsed_ms);

//         // 数据验证
//         const uint8_t *padded_bytes = reinterpret_cast<const uint8_t *>(data_ptr);
//         const uint8_t *decomp_bytes = reinterpret_cast<const uint8_t *>(host_decompressed_data);
//         const size_t decomp_size = device_base_column.n_values * sizeof(float);
//         if (memcmp(padded_bytes, decomp_bytes, decomp_size) != 0)
//         {
//             std::cout << "❌ 基础格式数据验证失败" << std::endl;
//             const float *p = data_ptr;
//             const float *d = host_decompressed_data;
//             int shown = 0;
//             for (size_t i = 0; i < device_base_column.n_values && shown < 10; ++i)
//             {
//                 if (std::abs(p[i] - d[i]) > 1e-10)
//                 {
//                     std::cout << "  不匹配[" << i << "]: " << p[i] << " vs " << d[i] << std::endl;
//                     ++shown;
//                 }
//             }
//         }
//         else
//         {
//             std::cout << "✓ 基础格式数据验证成功" << std::endl;
//         }

//         // 结果返回
//         const double comp_tp = compression_time_ms > 0.0 ? (original_size / (1024.0 * 1024.0 * 1024.0)) / (compression_time_ms / 1000.0) : 0.0;
//         const double decomp_tp = decompression_time_ms > 0.0 ? (original_size / (1024.0 * 1024.0 * 1024.0)) / (decompression_time_ms / 1000.0) : 0.0;

//         CompressionInfo result{
//             original_size / (1024.0 * 1024.0),
//             compressed_size / (1024.0 * 1024.0),
//             compression_ratio,
//             compression_kernel_time_ms,
//             compression_time_ms,
//             comp_tp,
//             decompression_kernel_time_ms,
//             decompression_time_ms,
//             decomp_tp};

//         delete[] host_decompressed_data;
//         flsgpu::host::free_column(device_base_column);
//         flsgpu::host::free_column(host_base_column);
//         cudaEventDestroy(kernel_start);
//         cudaEventDestroy(kernel_stop);
//         cudaDeviceSynchronize();
//         return result;
//     }
// }

// // ==================== 文件测试包装函数 ====================
// CompressionInfo test_compression(const std::string &file_path)
// {
//     std::vector<float> oriData = read_data_float(file_path);
//     return comp_ALP_G(oriData); // 调用扩展版实现
// }

// CompressionInfo test_beta_compression(const std::string &file_path, int beta)
// {
//     std::vector<float> oriData = read_data_float(file_path);
//     return comp_ALP_G(oriData);
// }

// // ==================== Google Test 用例(保持结构一致) ====================
// TEST(ALPGExtendedCompressorTest, CompressionDecompression)
// {
//     std::string dir_path = "../test/data/mew_tsbs";
//     bool warmup = false;
//     for (const auto &entry : fs::directory_iterator(dir_path))
//     {
//         if (entry.is_regular_file() && entry.path().extension() == ".csv")
//         {
//             std::string file_path = entry.path().string();
//             CompressionInfo result;
//             if (!warmup)
//             {
//                 test_compression(file_path);
//                 cudaDeviceSynchronize();
//                 warmup = true;
//             }
//             result = test_compression(file_path);
//             EXPECT_GT(result.compression_ratio, 0.0);
//             EXPECT_GT(result.comp_throughput, 0.0);
//             EXPECT_GT(result.decomp_throughput, 0.0);
//         }
//     }
// }

// // ==================== 主程序 ====================
// int main(int argc, char *argv[])
// {
//     cudaFree(0); // 初始化 CUDA

//     if (argc < 2)
//     {
//         ::testing::InitGoogleTest(&argc, argv);
//         return RUN_ALL_TESTS();
//     }

//     std::string arg = argv[1];

//     if (arg == "--dir" && argc >= 3)
//     {
//         std::string dir_path = argv[2];
//         std::cout << "📁 扩展格式处理目录: " << dir_path << std::endl;

//         std::vector<std::string> csv_files;
//         for (const auto &entry : fs::directory_iterator(dir_path))
//         {
//             if (entry.is_regular_file() && entry.path().extension() == ".csv")
//             {
//                 csv_files.push_back(entry.path().string());
//             }
//         }
//         if (csv_files.empty())
//         {
//             std::cerr << "❌ 未找到 CSV 文件" << std::endl;
//             return 1;
//         }

//         std::cout << "找到 " << csv_files.size() << " 个CSV文件" << std::endl;
//         std::cout << "\n=== 预热阶段(扩展) ===" << std::endl;
//         test_compression(csv_files[0]);
//         cudaDeviceSynchronize();

//         for (const auto &file_path : csv_files)
//         {
//             std::cout << "\n========================================" << std::endl;
//             std::cout << "文件: " << fs::path(file_path).filename() << std::endl;
//             std::cout << "========================================" << std::endl;

//             CompressionInfo total_result;
//             for (int i = 0; i < 3; ++i)
//             {
//                 std::cout << "\n--- 迭代 " << (i + 1) << " ---" << std::endl;
//                 CompressionInfo r = test_compression(file_path);
//                 total_result += r;
//                 cudaDeviceSynchronize();
//             }
//             total_result = total_result / 3;
//             total_result.print();
//         }
//         return 0;
//     }
//     else if (arg == "--file-beta" && argc >= 3)
//     {
//         std::string file_path = argv[2];
//         std::cout << "🔬 扩展格式 Beta 参数扫描: " << file_path << std::endl;
//         test_compression(file_path);
//         cudaDeviceSynchronize();
//         for (int beta = 4; beta <= 17; ++beta)
//         {
//             std::cout << "\n========================================" << std::endl;
//             std::cout << "Beta = " << beta << std::endl;
//             std::cout << "========================================" << std::endl;
//             CompressionInfo total_result;
//             for (int i = 0; i < 3; ++i)
//             {
//                 CompressionInfo r = test_beta_compression(file_path, beta);
//                 total_result += r;
//                 cudaDeviceSynchronize();
//             }
//             total_result = total_result / 3;
//             total_result.print();
//             return 0;
//         }
//     }
//     else
//     {
//         std::string file_path = arg;
//         std::cout << "📂 扩展格式处理文件: " << file_path << std::endl;
//         std::cout << "\n=== 预热(扩展) ===" << std::endl;
//         test_compression(file_path);
//         cudaDeviceSynchronize();

//         CompressionInfo total_result;
//         for (int i = 0; i < 3; ++i)
//         {
//             std::cout << "\n========================================" << std::endl;
//             std::cout << "迭代 " << (i + 1) << std::endl;
//             std::cout << "========================================" << std::endl;
//             CompressionInfo r = test_compression(file_path);
//             total_result += r;
//             cudaDeviceSynchronize();
//         }
//         total_result = total_result / 3;
//         total_result.print();
//         return 0;
//     }
// }

#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <filesystem>
#include <iomanip>
#include <numeric>
#include <cassert>

// ALP-G 头文件
#include "alp/alp-bindings.cuh"
#include "flsgpu/flsgpu-api.cuh"
#include "flsgpu/structs.cuh"
#include "data/dataset_utils.hpp"
#include "generated-bindings/kernel-bindings.cuh"
#include "engine/enums.cuh"
#include "engine/data.cuh"
#include "engine/verification.cuh"
#include "engine/device-utils.cuh"
#include "engine/kernels.cuh"

namespace fs = std::filesystem;

// 函数声明
CompressionInfo comp_ALP_G(std::vector<float> oriData);
CompressionInfo test_compression(const std::string& file_path);
CompressionInfo test_beta_compression(const std::string& file_path, int beta);

// ==================== 主压缩函数 ====================
CompressionInfo comp_ALP_G(std::vector<float> oriData) {
    const size_t original_num_elements = oriData.size();
    const size_t original_size = original_num_elements * sizeof(float);

    if (original_num_elements == 0) {
        std::cerr << "❌ 输入数据为空" << std::endl;
        return CompressionInfo{};
    }

    // ==================== 配置参数 ====================
    constexpr size_t VECTOR_SIZE = 1024;
    // 重要：UNPACK_N_VECTORS > 1 需要特殊的向量分组逻辑
    // 当 UNPACK_N_VECTORS = 4 时，GPU内核期望处理连续的4个向量组
    // 目前建议使用 UNPACK_N_VECTORS = 1 以确保正确性
    constexpr unsigned UNPACK_N_VECTORS = 1;  // 推荐值：1（安全），4（高性能但需要特殊处理）

    // 根据 ALP-G 源码中 FillWarpThreadblockMapping 的实际定义计算线程块参数
    // 对于 float 类型：
    // utils::get_n_lanes<float>() = 32
    // consts::THREADS_PER_WARP = 32
    // N_WARPS_PER_BLOCK = max(32/32, 2) = max(1, 2) = 2
    // N_THREADS_PER_BLOCK = 2 * 32 = 64
    // N_CONCURRENT_VECTORS_PER_BLOCK = 64 / 32 = 2
    constexpr size_t N_LANES_FLOAT = 32;
    constexpr size_t THREADS_PER_WARP = 32;
    constexpr size_t N_WARPS_PER_BLOCK = 2;  // max(32/32, 2) = 2
    constexpr size_t N_THREADS_PER_BLOCK = N_WARPS_PER_BLOCK * THREADS_PER_WARP;  // 2 * 32 = 64
    constexpr size_t N_CONCURRENT_VECTORS_PER_BLOCK = N_THREADS_PER_BLOCK / N_LANES_FLOAT;  // 64 / 32 = 2
    constexpr size_t VECTORS_PER_BLOCK = UNPACK_N_VECTORS * N_CONCURRENT_VECTORS_PER_BLOCK;  // 1 * 2 = 2

    // ==================== 数据填充策略 ====================
    size_t num_elements = original_num_elements;
    std::vector<float> paddedData;
    const float* data_ptr = oriData.data();

    // 检查数据可压缩性
    if (alp::is_compressable(data_ptr, num_elements)) {
        std::cout << "✓ 数据可压缩" << std::endl;
    } else {
        std::cout << "⚠️ 数据可压缩性较差" << std::endl;
    }

    // 计算需要的向量数
    size_t n_vecs = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;

    // 关键修复：确保向量数量能够被线程块完全处理
    // 每个线程块处理 VECTORS_PER_BLOCK 个向量，必须向上取整
    size_t n_vecs_padded = ((n_vecs + VECTORS_PER_BLOCK - 1) / VECTORS_PER_BLOCK) * VECTORS_PER_BLOCK;
    size_t num_elements_padded = n_vecs_padded * VECTOR_SIZE;

    if (num_elements_padded != original_num_elements) {
        size_t padding_needed = num_elements_padded - original_num_elements;
        num_elements = num_elements_padded;

        paddedData.reserve(num_elements);
        paddedData.insert(paddedData.end(), oriData.begin(), oriData.end());
        float padding_value = oriData.back();
        paddedData.insert(paddedData.end(), padding_needed, padding_value);
        data_ptr = paddedData.data();
    }

    // ==================== 压缩阶段 ====================
    auto start_total_compress = std::chrono::high_resolution_clock::now();
    flsgpu::host::ALPColumn<float> host_compressed_column;
    try {
        host_compressed_column = alp::encode<float>(data_ptr, num_elements, false);
    } catch (const std::exception& e) {
        std::cerr << "❌ ALP-G 压缩失败: " << e.what() << std::endl;
        return CompressionInfo{};
    }

    auto end_total_compress = std::chrono::high_resolution_clock::now();
    double compression_kernel_time = 0;
    double compression_total_time = std::chrono::duration<double, std::milli>(end_total_compress - start_total_compress).count();

    size_t compressed_size = host_compressed_column.compressed_size_bytes_alp;
    double compression_ratio = static_cast<double>(compressed_size) / original_size;

    if (compressed_size == 0) {
        std::cerr << "❌ 压缩失败: 压缩大小为0" << std::endl;
        flsgpu::host::free_column(host_compressed_column);
        return CompressionInfo{};
    }

    std::cout << "✓ 基础版压缩完成: " << compressed_size << " bytes, 比率="
              << compression_ratio << "x" << std::endl;

    // ==================== 解压阶段 ====================
    // 创建 CUDA 事件用于计时, 减少误差
    cudaEvent_t kernel_start{};
    cudaEvent_t kernel_stop{};
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    auto start_total_decompress = std::chrono::high_resolution_clock::now();

    // GPU 数据转移
    flsgpu::device::ALPColumn<float> device_column;
    try {
        device_column = host_compressed_column.copy_to_device();
        cudaDeviceSynchronize();
    } catch (const std::exception& e) {
        std::cerr << "❌ GPU 数据转移失败: " << e.what() << std::endl;
        flsgpu::host::free_column(host_compressed_column);
        return CompressionInfo{};
    }

    // 仅对 GPU 核函数本身计时（不含 D2H 拷贝）
    using ALPDecomp = flsgpu::device::ALPDecompressor<
        float, UNPACK_N_VECTORS,
        flsgpu::device::BitUnpackerStatefulBranchless<
            float, UNPACK_N_VECTORS, 1,
            flsgpu::device::ALPFunctor<float, UNPACK_N_VECTORS>>,
        flsgpu::device::StatefulALPExceptionPatcher<float, UNPACK_N_VECTORS, 1>,
        flsgpu::device::ALPColumn<float>>;

    const size_t n_vecs_mapping = utils::get_n_vecs_from_size(device_column.n_values);
    const ThreadblockMapping<float> mapping(UNPACK_N_VECTORS, n_vecs_mapping);

    // 分配 GPU 输出缓冲区
    float* d_out = nullptr;
    cudaMalloc(&d_out, num_elements * sizeof(float));

    float kernel_elapsed_ms = 0.0f;

    // 正式计时：仅核函数执行时间
    cudaEventRecord(kernel_start);
    kernels::device::decompress_column<float, UNPACK_N_VECTORS, 1,
                                       ALPDecomp,
                                       flsgpu::device::ALPColumn<float>>
        <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(device_column, d_out);
    cudaDeviceSynchronize();
    cudaEventRecord(kernel_stop);
    cudaEventSynchronize(kernel_stop);
    cudaEventElapsedTime(&kernel_elapsed_ms, kernel_start, kernel_stop);
    cudaDeviceSynchronize();

    // D2H 拷贝用于验证（不计入核函数时间）
    float* host_decompressed_data = new float[num_elements];
    cudaMemcpy(host_decompressed_data, d_out, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    auto end_total_decompress = std::chrono::high_resolution_clock::now();
    double decompression_kernel_time = static_cast<double>(kernel_elapsed_ms);
    double decompression_total_time = std::chrono::duration<double, std::milli>(end_total_decompress - start_total_decompress).count();

    // ==================== 数据验证 ====================
    const uint8_t* padded_bytes = reinterpret_cast<const uint8_t*>(data_ptr);
    const uint8_t* decompressed_bytes = reinterpret_cast<const uint8_t*>(host_decompressed_data);
    size_t actual_decomp_size = device_column.n_values * sizeof(float);

    if (memcmp(padded_bytes, decompressed_bytes, actual_decomp_size) != 0) {
        std::cout << "❌ 数据验证失败!" << std::endl;
        const float* padded_data = data_ptr;
        const float* decomp_data = host_decompressed_data;
        int error_count = 0;
        for (size_t i = 0; i < device_column.n_values && error_count < 10; ++i) {
            if (std::abs(padded_data[i] - decomp_data[i]) > 1e-10) {
                std::cout << "  数据不匹配 [" << i << "]: expected=" << padded_data[i]
                          << ", got=" << decomp_data[i] << std::endl;
                error_count++;
            }
        }
    } else {
        std::cout << "✓ 数据验证成功" << std::endl;
    }

    // ==================== 计算吞吐量 ====================
    double compression_total_throughput_gbps = (original_size / 1e9) / (compression_total_time / 1000.0);
    double decompression_total_throughput_gbps = (original_size / 1e9) / (decompression_total_time / 1000.0);

    CompressionInfo result = {
        original_size / (1024.0 * 1024.0),
        compressed_size / (1024.0 * 1024.0),
        compression_ratio,
        compression_kernel_time,
        compression_total_time,
        compression_total_throughput_gbps,
        decompression_kernel_time,
        decompression_total_time,
        decompression_total_throughput_gbps
    };

    // ==================== 清理资源 ====================
    delete[] host_decompressed_data;
    flsgpu::host::free_column(device_column);
    flsgpu::host::free_column(host_compressed_column);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    cudaDeviceSynchronize();

    return result;
}

// ==================== 文件测试包装函数 ====================
CompressionInfo test_compression(const std::string& file_path) {
    std::vector<float> oriData = read_data_float(file_path);
    return comp_ALP_G(oriData);
}

CompressionInfo test_beta_compression(const std::string& file_path, int beta) {
    (void)beta;
    std::vector<float> oriData = read_data_float(file_path);
    return comp_ALP_G(oriData);
}

// ==================== Google Test 测试用例 ====================
TEST(ALPGCompressorTest, CompressionDecompression) {
    std::string dir_path = "../test/data/mew_tsbs";
    bool warmup = false;

    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".csv") {
            std::string file_path = entry.path().string();

            CompressionInfo result;

            if (!warmup) {
                // 预热运行
                test_compression(file_path);
                cudaDeviceSynchronize();
                warmup = true;
            }

            // 正式测试
            result = test_compression(file_path);

            // 验证结果
            EXPECT_GT(result.compression_ratio, 0.0);
            EXPECT_GT(result.comp_throughput, 0.0);
            EXPECT_GT(result.decomp_throughput, 0.0);
        }
    }
}

int main(int argc, char* argv[]) {
    cudaFree(0);  // 初始化 CUDA

    if (argc < 2) {
        // 默认运行 Google Test
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }

    std::string arg = argv[1];

    if (arg == "--dir" && argc >= 3) {
        // 目录批处理模式
        std::string dir_path = argv[2];
        std::cout << "�� 处理目录: " << dir_path << std::endl;

        // 读取所有CSV文件
        std::vector<std::string> csv_files;
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".csv") {
                csv_files.push_back(entry.path().string());
            }
        }

        if (csv_files.empty()) {
            std::cerr << "❌ 未找到 CSV 文件" << std::endl;
            return 1;
        }

        std::cout << "找到 " << csv_files.size() << " 个CSV文件" << std::endl;

        // 预热
        std::cout << "\n=== 预热阶段 ===" << std::endl;
        test_compression(csv_files[0]);
        cudaDeviceSynchronize();

        // 对每个文件进行测试
        for (const auto& file_path : csv_files) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "文件: " << fs::path(file_path).filename() << std::endl;
            std::cout << "========================================" << std::endl;

            CompressionInfo total_result;

            // 3次迭代
            for (int i = 0; i < 3; ++i) {
                std::cout << "\n--- 迭代 " << (i + 1) << " ---" << std::endl;
                CompressionInfo result = test_compression(file_path);
                total_result += result;
                cudaDeviceSynchronize();
            }

            // 计算平均值
            total_result = total_result / 3;

            // 输出结果（模仿 LZ4 格式）
            total_result.print();
        }
        return 0;
    } else if (arg == "--file-beta" && argc >= 3) {
        // Beta 参数扫描模式
        std::string file_path = argv[2];
        std::cout << "�� Beta 参数扫描: " << file_path << std::endl;

        // 预热
        test_compression(file_path);
        cudaDeviceSynchronize();

        for (int beta = 4; beta <= 17; ++beta) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "Beta = " << beta << std::endl;
            std::cout << "========================================" << std::endl;

            CompressionInfo total_result;

            // 3次迭代
            for (int i = 0; i < 3; ++i) {
                CompressionInfo result = test_beta_compression(file_path, beta);
                total_result += result;
                cudaDeviceSynchronize();
            }

            // 计算平均值
            total_result = total_result / 3;

            // 输出结果
            total_result.print();
        }

        return 0;
    } else {
        // 单文件模式
        std::string file_path = arg;
        std::cout << "�� 处理文件: " << file_path << std::endl;

        // 预热
        std::cout << "\n=== 预热 ===" << std::endl;
        test_compression(file_path);
        cudaDeviceSynchronize();

        CompressionInfo total_result;

        // 3次迭代
        for (int i = 0; i < 3; ++i) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "迭代 " << (i + 1) << std::endl;
            std::cout << "========================================" << std::endl;

            CompressionInfo result = test_compression(file_path);
            total_result += result;
            cudaDeviceSynchronize();
        }

        // 计算平均值
        total_result = total_result / 3;
        total_result.print();
        return 0;
    }
}
