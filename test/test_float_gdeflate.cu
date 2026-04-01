#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#if __has_include(<nvcomp_12/nvcomp/gdeflate.hpp>)
#include <nvcomp_12/nvcomp/gdeflate.hpp>
#elif __has_include(<nvcomp_11/nvcomp/gdeflate.hpp>)
#include <nvcomp_11/nvcomp/gdeflate.hpp>
#else
#include <nvcomp/gdeflate.hpp>
#endif
#include <type_traits>
#include "data/dataset_utils.hpp"
#include <filesystem>
namespace fs = std::filesystem;

#define avg_times 3 

template <typename ManagerType>
ManagerType create_gdeflate_manager(
    size_t chunk_size,
    nvcompBatchedGdeflateOpts_t format_opts,
    cudaStream_t stream)
{
    if constexpr (std::is_constructible_v<ManagerType, size_t, const nvcompBatchedGdeflateOpts_t&, cudaStream_t>) {
        return ManagerType{chunk_size, format_opts, stream};
    } else if constexpr (std::is_constructible_v<ManagerType, size_t, nvcompBatchedGdeflateOpts_t, cudaStream_t>) {
        return ManagerType{chunk_size, format_opts, stream};
    } else {
        return ManagerType{chunk_size, static_cast<int>(format_opts.algo), stream};
    }
}

CompressionInfo test_compression(const std::string& file_path) {
    // Read input data
    std::vector<float> oriData = read_data_float(file_path);
    size_t in_bytes = oriData.size() * sizeof(float);
    if (in_bytes == 0) {
        std::cerr << "Error: Empty file or read failure: " << file_path << std::endl;
        return CompressionInfo{};
    }

    // Create CUDA stream and events
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaEvent_t start_event, end_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&end_event);

    // Allocate device memory
    uint8_t* device_input_data = nullptr;
    uint8_t* device_compressed_data = nullptr;
    uint8_t* device_decompressed_data = nullptr;
    
    cudaError_t err = cudaMalloc(&device_input_data, in_bytes);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating device_input_data: " << cudaGetErrorString(err) << std::endl;
        return CompressionInfo{};
    }

    // Create GDeflate manager
    const size_t chunk_size = 65536;
    nvcompBatchedGdeflateOpts_t format_opts = {0};
    auto manager = create_gdeflate_manager<nvcomp::GdeflateManager>(chunk_size, format_opts, stream);

    // Configure compression
    nvcomp::CompressionConfig comp_config = manager.configure_compression(in_bytes);
    err = cudaMalloc(&device_compressed_data, comp_config.max_compressed_buffer_size);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating device_compressed_data: " << cudaGetErrorString(err) << std::endl;
        cudaFree(device_input_data);
        return CompressionInfo{};
    }

    // =========================== Compression timing ===========================
    
    // 1. Measure H2D transfer time (compression)
    auto start_h2d_compress = std::chrono::high_resolution_clock::now();
    cudaEventRecord(start_event, stream);
    cudaMemcpyAsync(device_input_data, oriData.data(), in_bytes, cudaMemcpyHostToDevice, stream);
    cudaEventRecord(end_event, stream);
    cudaStreamSynchronize(stream);
    auto end_h2d_compress = std::chrono::high_resolution_clock::now();
    
    float h2d_compress_time_ms;
    cudaEventElapsedTime(&h2d_compress_time_ms, start_event, end_event);
    double h2d_compress_time = std::chrono::duration<double>(end_h2d_compress - start_h2d_compress).count();

    // 2. Measure compression kernel time
    auto start_compress_kernel = std::chrono::high_resolution_clock::now();
    cudaEventRecord(start_event, stream);
    manager.compress(device_input_data, device_compressed_data, comp_config);
    cudaEventRecord(end_event, stream);
    cudaStreamSynchronize(stream);
    auto end_compress_kernel = std::chrono::high_resolution_clock::now();
    
    float compress_kernel_time_ms;
    cudaEventElapsedTime(&compress_kernel_time_ms, start_event, end_event);
    double compress_kernel_time = std::chrono::duration<double>(end_compress_kernel - start_compress_kernel).count();

    // Get compressed size
    size_t comp_out_bytes = manager.get_compressed_output_size(device_compressed_data);

    // 3. Measure D2H transfer time (compressed result, optional)
    std::vector<uint8_t> compressed_host_data(comp_out_bytes);
    auto start_d2h_compress = std::chrono::high_resolution_clock::now();
    cudaEventRecord(start_event, stream);
    cudaMemcpyAsync(compressed_host_data.data(), device_compressed_data, comp_out_bytes, cudaMemcpyDeviceToHost, stream);
    cudaEventRecord(end_event, stream);
    cudaStreamSynchronize(stream);
    auto end_d2h_compress = std::chrono::high_resolution_clock::now();
    
    float d2h_compress_time_ms;
    cudaEventElapsedTime(&d2h_compress_time_ms, start_event, end_event);
    double d2h_compress_time = std::chrono::duration<double>(end_d2h_compress - start_d2h_compress).count();

    // Total compression time
    double total_compress_time = h2d_compress_time + compress_kernel_time + d2h_compress_time;

    // =========================== Decompression timing ===========================
    
    // Configure decompression
    nvcomp::DecompressionConfig decomp_config = manager.configure_decompression(device_compressed_data);
    err = cudaMalloc(&device_decompressed_data, decomp_config.decomp_data_size);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating device_decompressed_data: " << cudaGetErrorString(err) << std::endl;
        cudaFree(device_input_data);
        cudaFree(device_compressed_data);
        return CompressionInfo{};
    }
    
    // 1. Measure H2D transfer time (decompression, if compressed data is on host)
    auto start_h2d_decompress = std::chrono::high_resolution_clock::now();
    cudaEventRecord(start_event, stream);
    cudaMemcpyAsync(device_compressed_data, compressed_host_data.data(), comp_out_bytes, cudaMemcpyHostToDevice, stream);
    cudaEventRecord(end_event, stream);
    cudaStreamSynchronize(stream);
    auto end_h2d_decompress = std::chrono::high_resolution_clock::now();
    
    float h2d_decompress_time_ms;
    cudaEventElapsedTime(&h2d_decompress_time_ms, start_event, end_event);
    double h2d_decompress_time = std::chrono::duration<double>(end_h2d_decompress - start_h2d_decompress).count();

    // 2. Measure decompression kernel time
    auto start_decompress_kernel = std::chrono::high_resolution_clock::now();
    cudaEventRecord(start_event, stream);
    manager.decompress(device_decompressed_data, device_compressed_data, decomp_config);
    cudaEventRecord(end_event, stream);
    cudaStreamSynchronize(stream);
    auto end_decompress_kernel = std::chrono::high_resolution_clock::now();
    
    float decompress_kernel_time_ms;
    cudaEventElapsedTime(&decompress_kernel_time_ms, start_event, end_event);
    double decompress_kernel_time = std::chrono::duration<double>(end_decompress_kernel - start_decompress_kernel).count();

    // 3. Measure D2H transfer time (decompressed result)
    std::vector<float> decompressedData(oriData.size());
    auto start_d2h_decompress = std::chrono::high_resolution_clock::now();
    cudaEventRecord(start_event, stream);
    cudaMemcpyAsync(decompressedData.data(), device_decompressed_data, in_bytes, cudaMemcpyDeviceToHost, stream);
    cudaEventRecord(end_event, stream);
    cudaStreamSynchronize(stream);
    auto end_d2h_decompress = std::chrono::high_resolution_clock::now();
    
    float d2h_decompress_time_ms;
    cudaEventElapsedTime(&d2h_decompress_time_ms, start_event, end_event);
    double d2h_decompress_time = std::chrono::duration<double>(end_d2h_decompress - start_d2h_decompress).count();

    // Total decompression time
    double total_decompress_time = h2d_decompress_time + decompress_kernel_time + d2h_decompress_time;

    // =========================== Data validation ===========================
    const float tolerance = 1e-9;
    bool valid = true;
    for (size_t i = 0; i < oriData.size(); ++i) {
        if (std::abs(oriData[i] - decompressedData[i]) > tolerance) {
            valid = false;
            std::cerr << "Mismatch at position " << i 
                      << ": original=" << oriData[i] 
                      << ", decompressed=" << decompressedData[i] 
                      << std::endl;
            break;
        }
    }

    if (!valid) {
        std::cerr << "Data mismatch detected in " << file_path << "!" << std::endl;
        // FAIL() << "Data validation failed for file: " << file_path;
    } else {
        std::cout << "Decompression validated successfully." << std::endl;
    }

    // =========================== Performance metrics ===========================
    double compression_ratio = static_cast<double>(comp_out_bytes) / in_bytes; // Compression ratio (< 1)
    double data_size_gb = in_bytes / (1024.0 * 1024.0 * 1024.0);
    double compress_throughput = data_size_gb / (total_compress_time );
    double decompress_throughput = data_size_gb / (total_decompress_time);


    // =========================== Result aggregation ===========================
        CompressionInfo ans{
            in_bytes/1024.0/1024.0,
            static_cast<double>(comp_out_bytes) /1024.0/1024.0,
            compression_ratio,
            compress_kernel_time * 1000,
            total_compress_time * 1000 ,
            compress_throughput,
            decompress_kernel_time * 1000 ,
            total_decompress_time * 1000,
            decompress_throughput};

//     std::cout << "File: " << fs::path(file_path).filename() << std::endl;
// //    std::cout << "Compression Ratio: " << compression_ratio << std::endl;
//     printf("Compression Ratio: %0.3f\n",compression_ratio);
//     std::cout << "Total Compress Time: " << total_compress_time * 1000 << " ms" << std::endl;
//     std::cout << "Compress Kernel Time: " << compress_kernel_time * 1000 << " ms" << std::endl;
//     std::cout << "Total Decompress Time: " << total_decompress_time * 1000 << " ms" << std::endl;
//     std::cout << "Decompress Kernel Time: " << decompress_kernel_time * 1000 << " ms" << std::endl;
//     std::cout << "Data Size: " << data_size_gb << " GB" << std::endl;
//     std::cout << "Compression Throughput: " << compress_throughput << " GB/s" << std::endl;
//     std::cout << "Decompression Throughput: " << decompress_throughput << " GB/s" << std::endl;
    //translated comment
    
    if (device_input_data) {
        err = cudaFree(device_input_data);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing device_input_data: " << cudaGetErrorString(err) << std::endl;
        }
    }
    
    if (device_compressed_data) {
        err = cudaFree(device_compressed_data);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing device_compressed_data: " << cudaGetErrorString(err) << std::endl;
        }
    }
    
    if (device_decompressed_data) {
        err = cudaFree(device_decompressed_data);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing device_decompressed_data: " << cudaGetErrorString(err) << std::endl;
        }
    }
    
    // err = cudaEventDestroy(start_event);
    // if (err != cudaSuccess) {
    //     std::cerr << "Error destroying start_event: " << cudaGetErrorString(err) << std::endl;
    // }
    
    // err = cudaEventDestroy(end_event);
    // if (err != cudaSuccess) {
    //     std::cerr << "Error destroying end_event: " << cudaGetErrorString(err) << std::endl;
    // }
    
    // err = cudaStreamDestroy(stream);
    // if (err != cudaSuccess) {
    //     std::cerr << "Error destroying stream: " << cudaGetErrorString(err) << std::endl;
    // }
    return ans;
}
//Google Test
TEST(GDeflateCompressorTest, CompressionDecompression) {
    std::string dir_path = "../test/data/new_tsbs";
    if (!fs::exists(dir_path)) {
        GTEST_SKIP() << "Data directory not found: " << dir_path;
    }
    bool warm=0;
    int processed = 0;
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();
            CompressionInfo a;
            if(!warm)
            {
                // std::cout << "\n-------------------warm-------------------------- " << file_path << std::endl;
                test_compression(file_path);
                warm=1;
                // std::cout << "-------------------warm_end------------------------" << std::endl;
            }
            std::cout << "\nProcessing file: " << file_path << std::endl;
            for(int i=0;i<3;i++)
            {
                a+=test_compression(file_path);
            }
            a=a/3;
            a.print();
            std::cout << "---------------------------------------------" << std::endl;
            processed++;
        }
    }
    
    if (processed == 0) {
        std::cerr << "No files found in directory: " << dir_path << std::endl;
    }
}


int main(int argc, char *argv[]) {
    
    cudaFree(0);
    std::string arg = argv[1];
    
    if (arg == "--dir" && argc >= 3) {

        std::string dir_path = argv[2];

        //translated comment
        if (!fs::exists(dir_path)) {
            std::cerr << "指定的数据目录不存在: " << dir_path << std::endl;
            return 1;
        }

        bool warm=0;
        int processed = 0;
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                std::string file_path = entry.path().string();
                CompressionInfo a;
                if(!warm)
                {
                    // std::cout << "\n-------------------warm-------------------------- " << file_path << std::endl;
                    test_compression(file_path);
                    warm=1;
                    // std::cout << "-------------------warm_end------------------------" << std::endl;
                }
                std::cout << "\nProcessing file: " << file_path << std::endl;
                for(int i=0;i<3;i++)
                {
                    cudaDeviceReset();
                    a+=test_compression(file_path);
                }
                a=a/3;
                a.print();
                std::cout << "---------------------------------------------" << std::endl;
                processed++;
            }
        }
        
        if (processed == 0) {
            std::cerr << "No files found in directory: " << dir_path << std::endl;
        }
    }
    else{
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }
}