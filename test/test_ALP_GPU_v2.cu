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

// // ALP-G headers
// #include "alp/alp-bindings.cuh"
// #include "flsgpu/flsgpu-api.cuh"
// #include "flsgpu/structs.cuh"
// #include "data/dataset_utils.hpp"
// #include "generated-bindings/kernel-bindings.cuh"
// #include "engine/enums.cuh"
// #include "engine/data.cuh"
// #include "engine/verification.cuh"

// namespace fs = std::filesystem;

// // Function declaration (keep the same name as the basic version, internal extended format)
// CompressionInfo comp_ALP_G(std::vector<double> oriData);
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

// // ==================== Extended compression + GPU decompression ====================
// CompressionInfo comp_ALP_G(std::vector<double> oriData)
// {
//     const size_t original_num_elements = oriData.size();
//     const size_t original_size = original_num_elements * sizeof(double);
//     if (original_num_elements == 0)
//     {
//         std::cerr << "❌ Input data is empty" << std::endl;
//         return CompressionInfo{};
//     }

//     // Parameters aligned with the basic version
//     constexpr size_t VECTOR_SIZE = 1024;
//     constexpr unsigned UNPACK_N_VECTORS = 1; // Safe configuration
//     constexpr size_t N_LANES_DOUBLE = 16;
//     constexpr size_t THREADS_PER_WARP = 32;
//     constexpr size_t N_WARPS_PER_BLOCK = 2;
//     constexpr size_t N_THREADS_PER_BLOCK = N_WARPS_PER_BLOCK * THREADS_PER_WARP;
//     constexpr size_t N_CONCURRENT_VECTORS_PER_BLOCK = N_THREADS_PER_BLOCK / N_LANES_DOUBLE; // 64/16=4
//     constexpr size_t VECTORS_PER_BLOCK = UNPACK_N_VECTORS * N_CONCURRENT_VECTORS_PER_BLOCK; // 4

//     size_t num_elements = original_num_elements;
//     std::vector<double> paddedData;
//     const double *data_ptr = oriData.data();

//     // 1. Data preprocessing
//     // Pad to multiples of the block vector group size
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
//     // 2. Compression phase
//     auto encode_start = std::chrono::high_resolution_clock::now();
//     flsgpu::host::ALPColumn<double> host_base_column;
//     try
//     {
//         host_base_column = alp::encode<double>(data_ptr, num_elements, false);
//     }
//     catch (const std::exception &e)
//     {
//         std::cerr << "❌ Base compression failed: " << e.what() << std::endl;
//         return CompressionInfo{};
//     }
//     // Timing
//     auto encode_end = std::chrono::high_resolution_clock::now();
//     const double compression_time_ms = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();
//     constexpr double compression_kernel_time_ms = 0.0;
    
//     // Decide whether to use extended or base format
//     const bool use_extended = can_use_extended(host_base_column);
//     const size_t compressed_size = use_extended ? host_base_column.compressed_size_bytes_alp_extended
//                                                 : host_base_column.compressed_size_bytes_alp;
//     const double compression_ratio = static_cast<double>(compressed_size) / original_size;
//     std::cout << "✓ Compression done (" << (use_extended ? "extended" : "base") << "): "
//               << compressed_size << " bytes, ratio=" << compression_ratio << "x" << std::endl;

//     // Create CUDA events for timing
//     cudaEvent_t kernel_start{};
//     cudaEvent_t kernel_stop{};
//     cudaEventCreate(&kernel_start);
//     cudaEventCreate(&kernel_stop);
//     auto decomp_start = std::chrono::high_resolution_clock::now();
//     // Decompression
//     if (use_extended)
//     {

//         flsgpu::host::ALPExtendedColumn<double> host_extended_column = host_base_column.create_extended_column();
//         flsgpu::device::ALPExtendedColumn<double> device_extended_column;
//         // decomp_start = std::chrono::high_resolution_clock::now();
//translated comment
//         try
//         {
//             device_extended_column = host_extended_column.copy_to_device();
//             cudaDeviceSynchronize();
//         }
//         catch (const std::exception &e)
//         {
//             std::cerr << "❌ Copy of extended column to GPU failed: " << e.what() << std::endl;
//             flsgpu::host::free_column(host_extended_column);
//             flsgpu::host::free_column(host_base_column);
//             return CompressionInfo{};
//         }
        
//         float kernel_elapsed_ms = 0.0f;
//         double *host_decompressed_data = nullptr;
//translated comment
//         try
//         {
//             cudaEventRecord(kernel_start);
//             host_decompressed_data = bindings::decompress_column<double, flsgpu::device::ALPExtendedColumn<double>>(
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
//                 throw std::runtime_error("Extended GPU decompression returned nullptr");
//             }
//         }
//         catch (const std::exception &e)
//         {
//             std::cerr << "❌ Extended GPU decompression failed: " << e.what() << std::endl;
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

//         // Timing statistics
//         const double decompression_time_ms = std::chrono::duration<double, std::milli>(decomp_end - decomp_start).count();
//         const double decompression_kernel_time_ms = static_cast<double>(kernel_elapsed_ms);
        
//         // Verify decompressed data
//         const uint8_t *padded_bytes = reinterpret_cast<const uint8_t *>(data_ptr);
//         const uint8_t *decomp_bytes = reinterpret_cast<const uint8_t *>(host_decompressed_data);
//         const size_t decomp_size = device_extended_column.n_values * sizeof(double);
//         if (memcmp(padded_bytes, decomp_bytes, decomp_size) != 0)
//         {
//             std::cout << "❌ Extended format data verification failed" << std::endl;
//             const double *p = data_ptr;
//             const double *d = host_decompressed_data;
//             int shown = 0;
//             for (size_t i = 0; i < device_extended_column.n_values && shown < 10; ++i)
//             {
//                 if (std::abs(p[i] - d[i]) > 1e-10)
//                 {
//                     std::cout << "  Mismatch[" << i << "]: " << p[i] << " vs " << d[i] << std::endl;
//                     ++shown;
//                 }
//             }
//         }
//         else
//         {
//             std::cout << "✓ Extended format data verification succeeded" << std::endl;
//         }

//translated comment
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
//         // Data transfer
//         flsgpu::device::ALPColumn<double> device_base_column;
//         try
//         {
//             device_base_column = host_base_column.copy_to_device();
//             cudaDeviceSynchronize();
//         }
//         catch (const std::exception &e)
//         {
//             std::cerr << "❌ Copy of base column to GPU failed: " << e.what() << std::endl;
//             flsgpu::host::free_column(host_base_column);
//             return CompressionInfo{};
//         }

//         float kernel_elapsed_ms = 0.0f;
//         double *host_decompressed_data = nullptr;
//translated comment
//         try
//         {
//             cudaEventRecord(kernel_start);
//             host_decompressed_data = bindings::decompress_column<double, flsgpu::device::ALPColumn<double>>(
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
//throw std::runtime_error(" GPU nullptr");
//             }
//         }
//         catch (const std::exception &e)
//         {
//std::cerr << "❌ GPU : " << e.what() << std::endl;
//             if (host_decompressed_data)
//                 delete[] host_decompressed_data;
//             flsgpu::host::free_column(device_base_column);
//             flsgpu::host::free_column(host_base_column);
//             cudaEventDestroy(kernel_start);
//             cudaEventDestroy(kernel_stop);
//             return CompressionInfo{};
//         }
//translated comment
//         auto decomp_end = std::chrono::high_resolution_clock::now();
//         const double decompression_time_ms = std::chrono::duration<double, std::milli>(decomp_end - decomp_start).count();
//         const double decompression_kernel_time_ms = static_cast<double>(kernel_elapsed_ms);

//translated comment
//         const uint8_t *padded_bytes = reinterpret_cast<const uint8_t *>(data_ptr);
//         const uint8_t *decomp_bytes = reinterpret_cast<const uint8_t *>(host_decompressed_data);
//         const size_t decomp_size = device_base_column.n_values * sizeof(double);
//         if (memcmp(padded_bytes, decomp_bytes, decomp_size) != 0)
//         {
//std::cout << "❌ " << std::endl;
//             const double *p = data_ptr;
//             const double *d = host_decompressed_data;
//             int shown = 0;
//             for (size_t i = 0; i < device_base_column.n_values && shown < 10; ++i)
//             {
//                 if (std::abs(p[i] - d[i]) > 1e-10)
//                 {
//std::cout << " [" << i << "]: " << p[i] << " vs " << d[i] << std::endl;
//                     ++shown;
//                 }
//             }
//         }
//         else
//         {
//std::cout << "✓ " << std::endl;
//         }

//translated comment
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

//translated comment
// CompressionInfo test_compression(const std::string &file_path)
// {
//     std::vector<double> oriData = read_data(file_path);
//return comp_ALP_G(oriData); //
// }

// CompressionInfo test_beta_compression(const std::string &file_path, int beta)
// {
//     std::vector<double> oriData = read_data(file_path, beta);
//     return comp_ALP_G(oriData);
// }

//// ==================== Google Test ( ) ====================
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

//translated comment
// int main(int argc, char *argv[])
// {
//cudaFree(0); // CUDA

//     if (argc < 2)
//     {
//         ::testing::InitGoogleTest(&argc, argv);
//         return RUN_ALL_TESTS();
//     }

//     std::string arg = argv[1];

//     if (arg == "--dir" && argc >= 3)
//     {
//         std::string dir_path = argv[2];
//std::cout << "📁 : " << dir_path << std::endl;

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
//std::cerr << "❌ CSV " << std::endl;
//             return 1;
//         }

//std::cout << " " << csv_files.size() << " CSV " << std::endl;
//std::cout << "\n=== ( ) ===" << std::endl;
//         test_compression(csv_files[0]);
//         cudaDeviceSynchronize();

//         for (const auto &file_path : csv_files)
//         {
//             std::cout << "\n========================================" << std::endl;
//std::cout << " : " << fs::path(file_path).filename() << std::endl;
//             std::cout << "========================================" << std::endl;

//             CompressionInfo total_result;
//             for (int i = 0; i < 3; ++i)
//             {
//std::cout << "\n--- " << (i + 1) << " ---" << std::endl;
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
//std::cout << "🔬 Beta : " << file_path << std::endl;
//         // test_compression(file_path);
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
//             // return 0;
//         }
//     }
//     else
//     {
//         std::string file_path = arg;
//std::cout << "📂 : " << file_path << std::endl;
//std::cout << "\n=== ( ) ===" << std::endl;
//         test_compression(file_path);
//         cudaDeviceSynchronize();

//         CompressionInfo total_result;
//         for (int i = 0; i < 3; ++i)
//         {
//             std::cout << "\n========================================" << std::endl;
//std::cout << " " << (i + 1) << std::endl;
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




//translated comment
#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <array>
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

//ALP-G
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

//translated comment
CompressionInfo comp_ALP_G(std::vector<double> oriData);
CompressionInfo test_compression(const std::string &file_path);
CompressionInfo test_beta_compression(const std::string &file_path, int beta);

template <typename T>
bool can_use_extended(const flsgpu::host::ALPColumn<T> &column)
{
    constexpr size_t N_LANES = utils::get_n_lanes<T>();
    constexpr uint16_t MAX_LANE_COUNT = (1u << 6) - 1; // offsets_counts count field is 6 bits

    const size_t n_vecs = column.ffor.bp.get_n_vecs();
    for (size_t vec = 0; vec < n_vecs; ++vec)
    {
        const uint32_t exc_count = column.counts[vec];
        if (exc_count == 0)
        {
            continue;
        }

        const size_t exc_offset = column.exceptions_offsets[vec];
        std::array<uint16_t, N_LANES> lane_counts{};
        for (uint32_t i = 0; i < exc_count; ++i)
        {
            const uint16_t pos = column.positions[exc_offset + i];
            const uint16_t lane = static_cast<uint16_t>(pos % N_LANES);
            ++lane_counts[lane];
        }

        for (uint16_t lane_count : lane_counts)
        {
            if (lane_count > MAX_LANE_COUNT)
            {
                return false;
            }
        }
    }

    return true;
}

//==================== + GPU ====================
CompressionInfo comp_ALP_G(std::vector<double> oriData)
{
    const size_t original_num_elements = oriData.size();
    const size_t original_size = original_num_elements * sizeof(double);
    if (original_num_elements == 0)
    {
        std::cerr << "❌ 输入数据为空" << std::endl;
        return CompressionInfo{};
    }

    //translated comment
    constexpr size_t VECTOR_SIZE = 1024;
    constexpr unsigned UNPACK_N_VECTORS = 1; //translated comment
    constexpr size_t N_LANES_DOUBLE = 16;
    constexpr size_t THREADS_PER_WARP = 32;
    constexpr size_t N_WARPS_PER_BLOCK = 2;
    constexpr size_t N_THREADS_PER_BLOCK = N_WARPS_PER_BLOCK * THREADS_PER_WARP;
    constexpr size_t N_CONCURRENT_VECTORS_PER_BLOCK = N_THREADS_PER_BLOCK / N_LANES_DOUBLE; // 64/16=4
    constexpr size_t VECTORS_PER_BLOCK = UNPACK_N_VECTORS * N_CONCURRENT_VECTORS_PER_BLOCK; // 4

    size_t num_elements = original_num_elements;
    std::vector<double> paddedData;
    const double *data_ptr = oriData.data();

    //translated comment
    //translated comment
    size_t n_vecs = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
    size_t n_vecs_padded = ((n_vecs + VECTORS_PER_BLOCK - 1) / VECTORS_PER_BLOCK) * VECTORS_PER_BLOCK;
    size_t num_elements_padded = n_vecs_padded * VECTOR_SIZE;
    if (num_elements_padded != original_num_elements)
    {
        size_t pad_needed = num_elements_padded - original_num_elements;
        num_elements = num_elements_padded;
        paddedData.reserve(num_elements);
        paddedData.insert(paddedData.end(), oriData.begin(), oriData.end());
        paddedData.insert(paddedData.end(), pad_needed, oriData.back());
        data_ptr = paddedData.data();
    }
    //translated comment
    auto encode_start = std::chrono::high_resolution_clock::now();
    flsgpu::host::ALPColumn<double> host_base_column;
    try
    {
        host_base_column = alp::encode<double>(data_ptr, num_elements, false);
    }
    catch (const std::exception &e)
    {
        std::cerr << "❌ 基础压缩失败: " << e.what() << std::endl;
        return CompressionInfo{};
    }
    //translated comment
    auto encode_end = std::chrono::high_resolution_clock::now();
    const double compression_time_ms = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();
    constexpr double compression_kernel_time_ms = 0.0;
    
    //translated comment
    const bool use_extended = can_use_extended(host_base_column);
    const size_t compressed_size = use_extended ? host_base_column.compressed_size_bytes_alp_extended
                                                : host_base_column.compressed_size_bytes_alp;
    const double compression_ratio = static_cast<double>(compressed_size) / original_size;
    std::cout << "✓ 压缩完成(" << (use_extended ? "扩展" : "基础") << "): "
              << compressed_size << " bytes, 比率=" << compression_ratio << "x" << std::endl;

    //CUDA
    cudaEvent_t kernel_start{};
    cudaEvent_t kernel_stop{};
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    double format_prepare_time_ms = 0.0;
    auto decomp_start = std::chrono::high_resolution_clock::now();
    //translated comment
    if (use_extended)
    {

        auto format_prepare_start = std::chrono::high_resolution_clock::now();
        flsgpu::host::ALPExtendedColumn<double> host_extended_column = host_base_column.create_extended_column();
        auto format_prepare_end = std::chrono::high_resolution_clock::now();
        format_prepare_time_ms =
            std::chrono::duration<double, std::milli>(format_prepare_end - format_prepare_start).count();
        flsgpu::device::ALPExtendedColumn<double> device_extended_column;
        // decomp_start = std::chrono::high_resolution_clock::now();
        //translated comment
        try
        {
            device_extended_column = host_extended_column.copy_to_device();
            cudaDeviceSynchronize();
        }
        catch (const std::exception &e)
        {
            std::cerr << "❌ 扩展列复制到 GPU 失败: " << e.what() << std::endl;
            flsgpu::host::free_column(host_extended_column);
            flsgpu::host::free_column(host_base_column);
            return CompressionInfo{};
        }
        
        float kernel_elapsed_ms = 0.0f;
        double *host_decompressed_data = nullptr;
        //benchmark-compressors.cu GALP
        using GALPDecomp = flsgpu::device::ALPDecompressor<
            double, UNPACK_N_VECTORS,
            flsgpu::device::BitUnpackerStatefulBranchless<
                double, UNPACK_N_VECTORS, 1,
                flsgpu::device::ALPFunctor<double, UNPACK_N_VECTORS>>,
            flsgpu::device::PrefetchAllALPExceptionPatcher<double, UNPACK_N_VECTORS, 1>,
            flsgpu::device::ALPExtendedColumn<double>>;

        const size_t n_vecs_ext = utils::get_n_vecs_from_size(device_extended_column.n_values);
        const ThreadblockMapping<double> mapping(UNPACK_N_VECTORS, n_vecs_ext);

        //GPU （ CPU， D2H ）
        double *d_out = nullptr;
        cudaMalloc(&d_out, num_elements * sizeof(double));

        //translated comment
        // kernels::device::decompress_column<double, UNPACK_N_VECTORS, 1,
        //                                    GALPDecomp,
        //                                    flsgpu::device::ALPExtendedColumn<double>>
        //     <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(device_extended_column, d_out);
        // cudaDeviceSynchronize();

        //translated comment
        cudaEventRecord(kernel_start);
        kernels::device::decompress_column<double, UNPACK_N_VECTORS, 1,
                                           GALPDecomp,
                                           flsgpu::device::ALPExtendedColumn<double>>
            <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(device_extended_column, d_out);
        cudaDeviceSynchronize();
        cudaEventRecord(kernel_stop);
        cudaEventSynchronize(kernel_stop);
        cudaEventElapsedTime(&kernel_elapsed_ms, kernel_start, kernel_stop);
        cudaDeviceSynchronize();

        //translated comment
        host_decompressed_data = new double[num_elements];
        cudaMemcpy(host_decompressed_data, d_out, num_elements * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_out);

        auto decomp_end = std::chrono::high_resolution_clock::now();

        //translated comment
        const double decompression_time_ms = std::chrono::duration<double, std::milli>(decomp_end - decomp_start).count();
        const double decomp_h2d_decode_d2h_time_ms = std::max(0.0, decompression_time_ms - format_prepare_time_ms);
        const double decompression_kernel_time_ms = static_cast<double>(kernel_elapsed_ms);
        
        //translated comment
        const uint8_t *padded_bytes = reinterpret_cast<const uint8_t *>(data_ptr);
        const uint8_t *decomp_bytes = reinterpret_cast<const uint8_t *>(host_decompressed_data);
        const size_t decomp_size = device_extended_column.n_values * sizeof(double);
        if (memcmp(padded_bytes, decomp_bytes, decomp_size) != 0)
        {
            std::cout << "❌ 扩展格式数据验证失败" << std::endl;
            const double *p = data_ptr;
            const double *d = host_decompressed_data;
            int shown = 0;
            for (size_t i = 0; i < device_extended_column.n_values && shown < 10; ++i)
            {
                if (std::abs(p[i] - d[i]) > 1e-10)
                {
                    std::cout << "  不匹配[" << i << "]: " << p[i] << " vs " << d[i] << std::endl;
                    ++shown;
                }
            }
        }
        else
        {
            std::cout << "✓ 扩展格式数据验证成功" << std::endl;
        }

        //translated comment
        const double comp_tp = compression_time_ms > 0.0 ? (original_size / 1e9) / (compression_time_ms / 1000.0) : 0.0;
        const double decomp_tp = decompression_time_ms > 0.0 ? (original_size / 1e9) / (decompression_time_ms / 1000.0) : 0.0;
        const double decomp_tp_h2d_decode_d2h =
            decomp_h2d_decode_d2h_time_ms > 0.0 ? (original_size / 1e9) / (decomp_h2d_decode_d2h_time_ms / 1000.0) : 0.0;
        std::cout << "  格式准备时间(format_prepare): " << format_prepare_time_ms << " ms" << std::endl;
        std::cout << "  解压时间(H2D+Decode+D2H): " << decomp_h2d_decode_d2h_time_ms << " ms" << std::endl;
        std::cout << "  解压吞吐量(H2D+Decode+D2H): " << decomp_tp_h2d_decode_d2h << " GB/s" << std::endl;

        CompressionInfo result{
            original_size / (1024.0 * 1024.0),
            compressed_size / (1024.0 * 1024.0),
            compression_ratio,
            compression_kernel_time_ms,
            compression_time_ms,
            comp_tp,
            decompression_kernel_time_ms,
            decompression_time_ms,
            decomp_tp};

        delete[] host_decompressed_data;
        flsgpu::host::free_column(device_extended_column);
        flsgpu::host::free_column(host_extended_column);
        flsgpu::host::free_column(host_base_column);
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_stop);
        cudaDeviceSynchronize();
        return result;
    }
    else{
        //translated comment
        flsgpu::device::ALPColumn<double> device_base_column;
        try
        {
            device_base_column = host_base_column.copy_to_device();
            cudaDeviceSynchronize();
        }
        catch (const std::exception &e)
        {
            std::cerr << "❌ 基础列复制到 GPU 失败: " << e.what() << std::endl;
            flsgpu::host::free_column(host_base_column);
            return CompressionInfo{};
        }

        float kernel_elapsed_ms = 0.0f;
        double *host_decompressed_data = nullptr;
        //benchmark-compressors.cu ALP
        using ALPDecomp = flsgpu::device::ALPDecompressor<
            double, UNPACK_N_VECTORS,
            flsgpu::device::BitUnpackerStatefulBranchless<
                double, UNPACK_N_VECTORS, 1,
                flsgpu::device::ALPFunctor<double, UNPACK_N_VECTORS>>,
            flsgpu::device::StatefulALPExceptionPatcher<double, UNPACK_N_VECTORS, 1>,
            flsgpu::device::ALPColumn<double>>;

        const size_t n_vecs_base = utils::get_n_vecs_from_size(device_base_column.n_values);
        const ThreadblockMapping<double> mapping(UNPACK_N_VECTORS, n_vecs_base);

        //GPU （ CPU， D2H ）
        double *d_out = nullptr;
        cudaMalloc(&d_out, num_elements * sizeof(double));

        //translated comment
        // kernels::device::decompress_column<double, UNPACK_N_VECTORS, 1,
        //                                    ALPDecomp,
        //                                    flsgpu::device::ALPColumn<double>>
        //     <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(device_base_column, d_out);
        // cudaDeviceSynchronize();

        //translated comment
        cudaEventRecord(kernel_start);
        kernels::device::decompress_column<double, UNPACK_N_VECTORS, 1,
                                           ALPDecomp,
                                           flsgpu::device::ALPColumn<double>>
            <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(device_base_column, d_out);
        cudaDeviceSynchronize();
        cudaEventRecord(kernel_stop);
        cudaEventSynchronize(kernel_stop);
        cudaEventElapsedTime(&kernel_elapsed_ms, kernel_start, kernel_stop);
        cudaDeviceSynchronize();

        //translated comment
        host_decompressed_data = new double[num_elements];
        cudaMemcpy(host_decompressed_data, d_out, num_elements * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_out);
        //translated comment
        auto decomp_end = std::chrono::high_resolution_clock::now();
        const double decompression_time_ms = std::chrono::duration<double, std::milli>(decomp_end - decomp_start).count();
        const double decompression_kernel_time_ms = static_cast<double>(kernel_elapsed_ms);

        //translated comment
        const uint8_t *padded_bytes = reinterpret_cast<const uint8_t *>(data_ptr);
        const uint8_t *decomp_bytes = reinterpret_cast<const uint8_t *>(host_decompressed_data);
        const size_t decomp_size = device_base_column.n_values * sizeof(double);
        if (memcmp(padded_bytes, decomp_bytes, decomp_size) != 0)
        {
            std::cout << "❌ 基础格式数据验证失败" << std::endl;
            const double *p = data_ptr;
            const double *d = host_decompressed_data;
            int shown = 0;
            for (size_t i = 0; i < device_base_column.n_values && shown < 10; ++i)
            {
                if (std::abs(p[i] - d[i]) > 1e-10)
                {
                    std::cout << "  不匹配[" << i << "]: " << p[i] << " vs " << d[i] << std::endl;
                    ++shown;
                }
            }
        }
        else
        {
            std::cout << "✓ 基础格式数据验证成功" << std::endl;
        }

        //translated comment
        const double comp_tp = compression_time_ms > 0.0 ? (original_size / 1e9) / (compression_time_ms / 1000.0) : 0.0;
        const double decomp_tp = decompression_time_ms > 0.0 ? (original_size / 1e9) / (decompression_time_ms / 1000.0) : 0.0;
        std::cout << "  格式准备时间(format_prepare): 0 ms" << std::endl;
        std::cout << "  解压时间(H2D+Decode+D2H): " << decompression_time_ms << " ms" << std::endl;
        std::cout << "  解压吞吐量(H2D+Decode+D2H): " << decomp_tp << " GB/s" << std::endl;

        CompressionInfo result{
            original_size / (1024.0 * 1024.0),
            compressed_size / (1024.0 * 1024.0),
            compression_ratio,
            compression_kernel_time_ms,
            compression_time_ms,
            comp_tp,
            decompression_kernel_time_ms,
            decompression_time_ms,
            decomp_tp};

        delete[] host_decompressed_data;
        flsgpu::host::free_column(device_base_column);
        flsgpu::host::free_column(host_base_column);
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_stop);
        cudaDeviceSynchronize();
        return result;
    }
}

//translated comment
CompressionInfo test_compression(const std::string &file_path)
{
    std::vector<double> oriData = read_data(file_path);
    return comp_ALP_G(oriData); //translated comment
}

CompressionInfo test_beta_compression(const std::string &file_path, int beta)
{
    std::vector<double> oriData = read_data(file_path, beta);
    return comp_ALP_G(oriData);
}

//==================== Google Test ( ) ====================
TEST(ALPGExtendedCompressorTest, CompressionDecompression)
{
    std::string dir_path = "../test/data/mew_tsbs";
    bool warmup = false;
    for (const auto &entry : fs::directory_iterator(dir_path))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".csv")
        {
            std::string file_path = entry.path().string();
            CompressionInfo result;
            if (!warmup)
            {
                test_compression(file_path);
                cudaDeviceSynchronize();
                warmup = true;
            }
            result = test_compression(file_path);
            EXPECT_GT(result.compression_ratio, 0.0);
            EXPECT_GT(result.comp_throughput, 0.0);
            EXPECT_GT(result.decomp_throughput, 0.0);
        }
    }
}

//translated comment
int main(int argc, char *argv[])
{
    cudaFree(0); //CUDA

    if (argc < 2)
    {
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }

    std::string arg = argv[1];

    if (arg == "--dir" && argc >= 3)
    {
        std::string dir_path = argv[2];
        std::cout << "�� 扩展格式处理目录: " << dir_path << std::endl;

        std::vector<std::string> csv_files;
        for (const auto &entry : fs::directory_iterator(dir_path))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".csv")
            {
                csv_files.push_back(entry.path().string());
            }
        }
        if (csv_files.empty())
        {
            std::cerr << "❌ 未找到 CSV 文件" << std::endl;
            return 1;
        }

        std::cout << "找到 " << csv_files.size() << " 个CSV文件" << std::endl;
        std::cout << "\n=== 预热阶段(扩展) ===" << std::endl;
        test_compression(csv_files[0]);
        cudaDeviceSynchronize();

        for (const auto &file_path : csv_files)
        {
            std::cout << "\n========================================" << std::endl;
            std::cout << "文件: " << fs::path(file_path).filename() << std::endl;
            std::cout << "========================================" << std::endl;

            CompressionInfo total_result;
            for (int i = 0; i < 3; ++i)
            {
                std::cout << "\n--- 迭代 " << (i + 1) << " ---" << std::endl;
                CompressionInfo r = test_compression(file_path);
                total_result += r;
                cudaDeviceSynchronize();
            }
            total_result = total_result / 3;
            total_result.print();
        }
        return 0;
    }
    else if (arg == "--file-beta" && argc >= 3)
    {
        std::string file_path = argv[2];
        std::cout << "�� 扩展格式 Beta 参数扫描: " << file_path << std::endl;
        test_compression(file_path);
        cudaDeviceSynchronize();
        for (int beta = 4; beta <= 17; ++beta)
        {
            std::cout << "\n========================================" << std::endl;
            std::cout << "Beta = " << beta << std::endl;
            std::cout << "========================================" << std::endl;
            CompressionInfo total_result;
            for (int i = 0; i < 3; ++i)
            {
                CompressionInfo r = test_beta_compression(file_path, beta);
                total_result += r;
                cudaDeviceSynchronize();
            }
            total_result = total_result / 3;
            total_result.print();
            return 0;
        }
    }
    else
    {
        std::string file_path = arg;
        std::cout << "�� 扩展格式处理文件: " << file_path << std::endl;
        std::cout << "\n=== 预热(扩展) ===" << std::endl;
        test_compression(file_path);
        cudaDeviceSynchronize();

        CompressionInfo total_result;
        for (int i = 0; i < 3; ++i)
        {
            std::cout << "\n========================================" << std::endl;
            std::cout << "迭代 " << (i + 1) << std::endl;
            std::cout << "========================================" << std::endl;
            CompressionInfo r = test_compression(file_path);
            total_result += r;
            cudaDeviceSynchronize();
        }
        total_result = total_result / 3;
        total_result.print();
        return 0;
    }
}