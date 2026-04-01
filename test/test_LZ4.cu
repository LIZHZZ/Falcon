
#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <iostream>
#include <nvcomp/lz4.hpp>  //nvcomp LZ4
#include <cassert>
#include <nvcomp/lz4.h>
#include <cuda_runtime.h>
#include "data/dataset_utils.hpp"
#include <filesystem>
namespace fs = std::filesystem;
CompressionInfo comp_LZ4(std::vector<double> oriData);
CompressionInfo test_compression(const std::string& file_path) {
    //translated comment
    std::vector<double> oriData = read_data(file_path);
    return comp_LZ4(oriData);
}

CompressionInfo test_beta_compression(const std::string& file_path,int beta) {
    //translated comment
    std::vector<double> oriData = read_data(file_path,beta);
    return comp_LZ4(oriData);
}

CompressionInfo comp_LZ4(std::vector<double> oriData)
{
    //translated comment
    size_t in_bytes = oriData.size() * sizeof(double);

    //char ， LZ4
    char* input_data = reinterpret_cast<char*>(oriData.data());
    
    //CUDA
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //CUDA
    cudaEvent_t start_event, end_event;
    cudaEvent_t compress_h2d_start, compress_h2d_end;
    cudaEvent_t compress_kernel_start, compress_kernel_end;
    cudaEvent_t compress_d2h_start, compress_d2h_end;
    cudaEvent_t decompress_h2d_start, decompress_h2d_end;
    cudaEvent_t decompress_kernel_start, decompress_kernel_end;
    cudaEvent_t decompress_d2h_start, decompress_d2h_end;

    cudaEventCreate(&start_event);
    cudaEventCreate(&end_event);
    cudaEventCreate(&compress_h2d_start);
    cudaEventCreate(&compress_h2d_end);
    cudaEventCreate(&compress_kernel_start);
    cudaEventCreate(&compress_kernel_end);
    cudaEventCreate(&compress_d2h_start);
    cudaEventCreate(&compress_d2h_end);
    cudaEventCreate(&decompress_h2d_start);
    cudaEventCreate(&decompress_h2d_end);
    cudaEventCreate(&decompress_kernel_start);
    cudaEventCreate(&decompress_kernel_end);
    cudaEventCreate(&decompress_d2h_start);
    cudaEventCreate(&decompress_d2h_end);

    //translated comment
    const size_t chunk_size = 65536;
    const size_t batch_size = (in_bytes + chunk_size - 1) / chunk_size;

    //CUDA
    char* device_input_data;
    cudaMalloc(&device_input_data, in_bytes);

    size_t* host_uncompressed_bytes;
    cudaMallocHost(&host_uncompressed_bytes, sizeof(size_t) * batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        if (i + 1 < batch_size) {
            host_uncompressed_bytes[i] = chunk_size;
        } else {
            host_uncompressed_bytes[i] = in_bytes - (chunk_size * i);
        }
    }

    //translated comment
    void** host_uncompressed_ptrs;
    cudaMallocHost(&host_uncompressed_ptrs, sizeof(void*) * batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        host_uncompressed_ptrs[i] = device_input_data + chunk_size * i;
    }

    size_t* device_uncompressed_bytes;
    void** device_uncompressed_ptrs;
    cudaMalloc(&device_uncompressed_bytes, sizeof(size_t) * batch_size);
    cudaMalloc(&device_uncompressed_ptrs, sizeof(void*) * batch_size);

    //translated comment
    size_t temp_bytes;
    nvcompBatchedLZ4CompressGetTempSize(batch_size, chunk_size, nvcompBatchedLZ4DefaultOpts, &temp_bytes);
    void* device_temp_ptr;
    cudaMalloc(&device_temp_ptr, temp_bytes);

    //translated comment
    size_t max_out_bytes;
    nvcompBatchedLZ4CompressGetMaxOutputChunkSize(chunk_size, nvcompBatchedLZ4DefaultOpts, &max_out_bytes);

    //translated comment
    void** host_compressed_ptrs;
    cudaMallocHost(&host_compressed_ptrs, sizeof(void*) * batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        cudaMalloc(&host_compressed_ptrs[i], max_out_bytes);
    }

    void** device_compressed_ptrs;
    cudaMalloc(&device_compressed_ptrs, sizeof(void*) * batch_size);

    size_t* device_compressed_bytes;
    cudaMalloc(&device_compressed_bytes, sizeof(size_t) * batch_size);

    //translated comment
    // std::cout << "=== Compression Phase ===" << std::endl;
    
    //translated comment
    cudaEventRecord(start_event, stream);
    
    //H2D: Host to Device
    cudaEventRecord(compress_h2d_start, stream);
    
    cudaMemcpyAsync(device_input_data, input_data, in_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_uncompressed_bytes, host_uncompressed_bytes, sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_uncompressed_ptrs, host_uncompressed_ptrs, sizeof(void*) * batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_compressed_ptrs, host_compressed_ptrs, sizeof(void*) * batch_size, cudaMemcpyHostToDevice, stream);
    
    cudaEventRecord(compress_h2d_end, stream);

    //KERNEL:
    cudaEventRecord(compress_kernel_start, stream);
    
    //translated comment
    nvcompStatus_t comp_res = nvcompBatchedLZ4CompressAsync(
        device_uncompressed_ptrs,
        device_uncompressed_bytes,
        chunk_size,
        batch_size,
        device_temp_ptr,
        temp_bytes,
        device_compressed_ptrs,
        device_compressed_bytes,
        nvcompBatchedLZ4DefaultOpts,
        stream);
    cudaDeviceSynchronize();
    if (comp_res != nvcompSuccess) {
        std::cerr << "Compression failed!" << std::endl;
        assert(comp_res == nvcompSuccess);
    }
    
    cudaEventRecord(compress_kernel_end, stream);

    //D2H: Device to Host
    cudaEventRecord(compress_d2h_start, stream);
    
    size_t* host_compressed_bytes = new size_t[batch_size];
    cudaMemcpyAsync(host_compressed_bytes, device_compressed_bytes, 
                   sizeof(size_t) * batch_size, cudaMemcpyDeviceToHost, stream);
    
    //translated comment
    char** host_compressed_data = new char*[batch_size];
    for (size_t i = 0; i < batch_size; ++i) {
        host_compressed_data[i] = new char[max_out_bytes];
        cudaMemcpyAsync(host_compressed_data[i], host_compressed_ptrs[i], 
                       max_out_bytes, cudaMemcpyDeviceToHost, stream);
    }
    
    cudaEventRecord(compress_d2h_end, stream);
    
    //translated comment
    cudaStreamSynchronize(stream);
    
    float compress_h2d_time, compress_kernel_time, compress_d2h_time;
    cudaEventElapsedTime(&compress_h2d_time, compress_h2d_start, compress_h2d_end);
    cudaEventElapsedTime(&compress_kernel_time, compress_kernel_start, compress_kernel_end);
    cudaEventElapsedTime(&compress_d2h_time, compress_d2h_start, compress_d2h_end);
    
    float total_compress_time = compress_h2d_time + compress_kernel_time + compress_d2h_time;


    //translated comment
    size_t total_compressed = 0;
    for (size_t i = 0; i < batch_size; ++i) {
        total_compressed += host_compressed_bytes[i];
    }
    double compression_ratio = total_compressed / static_cast<double>(in_bytes);

    //translated comment
    // std::cout << "\n=== Decompression Phase ===" << std::endl;
    
    //translated comment
    char* device_output_data;
    cudaMalloc(&device_output_data, in_bytes);
    
    //translated comment
    void** host_decompressed_ptrs;
    cudaMallocHost(&host_decompressed_ptrs, sizeof(void*) * batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        host_decompressed_ptrs[i] = device_output_data + chunk_size * i;
    }
    
    void** device_decompressed_ptrs;
    cudaMalloc(&device_decompressed_ptrs, sizeof(void*) * batch_size);

    //translated comment
    size_t decomp_temp_bytes;
    nvcompBatchedLZ4DecompressGetTempSize(batch_size, chunk_size, &decomp_temp_bytes);
    void* device_decomp_temp;
    cudaMalloc(&device_decomp_temp, decomp_temp_bytes);

    nvcompStatus_t* device_statuses;
    cudaMalloc(&device_statuses, sizeof(nvcompStatus_t) * batch_size);

    size_t* device_actual_uncompressed_bytes;
    cudaMalloc(&device_actual_uncompressed_bytes, sizeof(size_t) * batch_size);

    //translated comment
    cudaEventRecord(decompress_h2d_start, stream);
    
    //translated comment
    for (size_t i = 0; i < batch_size; ++i) {
        cudaMemcpyAsync(host_compressed_ptrs[i], host_compressed_data[i], 
                       host_compressed_bytes[i], cudaMemcpyHostToDevice, stream);
    }
    
    cudaMemcpyAsync(device_compressed_bytes, host_compressed_bytes, 
                   sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_compressed_ptrs, host_compressed_ptrs, 
                   sizeof(void*) * batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_decompressed_ptrs, host_decompressed_ptrs, 
                   sizeof(void*) * batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_uncompressed_bytes, host_uncompressed_bytes, 
                   sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);
    
    cudaEventRecord(decompress_h2d_end, stream);

    //KERNEL:
    cudaEventRecord(decompress_kernel_start, stream);
    
    //translated comment
    nvcompStatus_t decomp_res = nvcompBatchedLZ4DecompressAsync(
        device_compressed_ptrs,
        device_compressed_bytes,
        device_uncompressed_bytes,
        device_actual_uncompressed_bytes,
        batch_size,
        device_decomp_temp,
        decomp_temp_bytes,
        device_decompressed_ptrs,
        device_statuses,
        stream);
    cudaDeviceSynchronize();
    if (decomp_res != nvcompSuccess) {
        std::cerr << "Decompression failed!" << std::endl;
        assert(decomp_res == nvcompSuccess);
    }

    cudaEventRecord(decompress_kernel_end, stream);

    //translated comment
    cudaEventRecord(decompress_d2h_start, stream);
    
    char* reconstructed = new char[in_bytes];
    cudaMemcpyAsync(reconstructed, device_output_data, in_bytes, 
                   cudaMemcpyDeviceToHost, stream);
    
    cudaEventRecord(decompress_d2h_end, stream);
    cudaEventRecord(end_event, stream);

    //translated comment
    cudaStreamSynchronize(stream);
    
    float decompress_h2d_time, decompress_kernel_time, decompress_d2h_time;
    cudaEventElapsedTime(&decompress_h2d_time, decompress_h2d_start, decompress_h2d_end);
    cudaEventElapsedTime(&decompress_kernel_time, decompress_kernel_start, decompress_kernel_end);
    cudaEventElapsedTime(&decompress_d2h_time, decompress_d2h_start, decompress_d2h_end);
    
    float total_decompress_time = decompress_h2d_time + decompress_kernel_time + decompress_d2h_time;
    float total_overall_time;
    cudaEventElapsedTime(&total_overall_time, start_event, end_event);
    //translated comment
    double data_size_gb = in_bytes / (1024.0 * 1024.0 * 1024.0);
    double compress_throughput = data_size_gb / (total_compress_time / 1000.0);
    double decompress_throughput = data_size_gb / (total_decompress_time / 1000.0);

    CompressionInfo a{
        in_bytes/1024.0/1024.0,
        total_compressed/1024.0/1024.0,
        compression_ratio,
        compress_kernel_time,
        total_compress_time,
        compress_throughput,
        decompress_kernel_time,
        total_decompress_time,
        decompress_throughput

    };

    //translated comment
    if (memcmp(input_data, reconstructed, in_bytes) != 0) {
        std::cout << "Data mismatch!" << std::endl;
    } else {
        // std::cout << "\nData verification: PASSED" << std::endl;
    }

    //translated comment
    delete[] host_compressed_bytes;
    for (size_t i = 0; i < batch_size; ++i) {
        delete[] host_compressed_data[i];
    }
    delete[] host_compressed_data;
    delete[] reconstructed;

    //CUDA
    cudaEventDestroy(start_event);
    cudaEventDestroy(end_event);
    cudaEventDestroy(compress_h2d_start);
    cudaEventDestroy(compress_h2d_end);
    cudaEventDestroy(compress_kernel_start);
    cudaEventDestroy(compress_kernel_end);
    cudaEventDestroy(compress_d2h_start);
    cudaEventDestroy(compress_d2h_end);
    cudaEventDestroy(decompress_h2d_start);
    cudaEventDestroy(decompress_h2d_end);
    cudaEventDestroy(decompress_kernel_start);
    cudaEventDestroy(decompress_kernel_end);
    cudaEventDestroy(decompress_d2h_start);
    cudaEventDestroy(decompress_d2h_end);

    //CUDA
    cudaStreamSynchronize(stream);
    cudaFree(device_input_data);
    cudaFree(device_output_data);
    cudaFree(device_uncompressed_bytes);
    cudaFree(device_uncompressed_ptrs);
    cudaFree(device_temp_ptr);
    cudaFree(device_compressed_ptrs);
    cudaFree(device_compressed_bytes);
    cudaFree(device_decomp_temp);
    cudaFree(device_statuses);
    cudaFree(device_actual_uncompressed_bytes);
    cudaFree(device_decompressed_ptrs);
    
    cudaFreeHost(host_uncompressed_bytes);
    cudaFreeHost(host_uncompressed_ptrs);
    cudaFreeHost(host_compressed_ptrs);
    cudaFreeHost(host_decompressed_ptrs);
    
    cudaStreamDestroy(stream);
    return a;
}

//Google Test
TEST(LZ4CompressorTest, CompressionDecompression) {
    //translated comment
    // std::string dir_path = "../test/data/big"; 
    std::string dir_path = "../test/data/mew_tsbs"; 
    bool warmup = 0;

    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();
            if(!warmup)
            {
                warmup=1;
                std::cout << "====================warmup==========================" << std::endl;
                test_compression(file_path);
                std::cout << "====================warmup_end=========================" << std::endl;
            }
            std::cout << "正在处理文件: " << file_path << std::endl;
            test_compression(file_path);
            std::cout << "==============================================" << std::endl;
        }
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
    else if (arg == "--file-beta" && argc >= 3) {

        std::string file_path = argv[2];

        for(int beta=4;beta<18;beta++)
        {
            std::cout << "\nProcessing file: " << file_path;

            printf("beta:%d\n",beta);
            CompressionInfo a;
            // for(int i=0;i<3;i++)
            // {
                cudaDeviceReset();
                a+=test_beta_compression(file_path,beta);
            // }
            // a=a/3;
            a.print();
            std::cout << "---------------------------------------------" << std::endl;
        }

    }
    else{
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }
}