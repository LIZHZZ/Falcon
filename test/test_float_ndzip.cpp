#include "ndzip/cuda.hh"
#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <cuda_runtime.h>
#include <thread>
#include "data/dataset_utils.hpp"
namespace fs = std::filesystem;

//translated comment
std::vector<float> read_data1(const std::string &file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + file_path);
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg() / sizeof(float);
    file.seekg(0, std::ios::beg);

    std::vector<float> data(size);
    file.read(reinterpret_cast<char *>(data.data()), size * sizeof(float));
    file.close();
    return data;
}
void test_cuda_compression(std::vector<float> oriData ,CompressionInfo &ans) ;
//CUDA
CompressionInfo test_compression(const std::string &file_path) {
    //translated comment
    std::vector<float> oriData = read_data1(file_path);
    CompressionInfo ans;
    test_cuda_compression(oriData,ans);
    return ans;
}

void test_cuda_compression(std::vector<float> oriData ,CompressionInfo &ans) {
    //translated comment
    size_t data_size = oriData.size();
    ndzip::extent data_extent{static_cast<ndzip::index_type>(data_size)};
    
    //translated comment
    size_t original_size = oriData.size() * sizeof(float);
    
    //CUDA
    cudaEvent_t start_total, end_total;
    cudaEvent_t start_h2d, end_h2d;
    cudaEvent_t start_compress, end_compress;
    cudaEvent_t start_d2h_compress, end_d2h_compress;
    cudaEvent_t start_h2d_decompress, end_h2d_decompress;
    cudaEvent_t start_decompress, end_decompress;
    cudaEvent_t start_d2h_decompress, end_d2h_decompress;
    
    cudaEventCreate(&start_total);
    cudaEventCreate(&end_total);
    cudaEventCreate(&start_h2d);
    cudaEventCreate(&end_h2d);
    cudaEventCreate(&start_compress);
    cudaEventCreate(&end_compress);
    cudaEventCreate(&start_d2h_compress);
    cudaEventCreate(&end_d2h_compress);
    cudaEventCreate(&start_h2d_decompress);
    cudaEventCreate(&end_h2d_decompress);
    cudaEventCreate(&start_decompress);
    cudaEventCreate(&end_decompress);
    cudaEventCreate(&start_d2h_decompress);
    cudaEventCreate(&end_d2h_decompress);

    //translated comment
    cudaEventRecord(start_total);
    
    //translated comment
    //std::cout << "CUDA \n";
    
    //1. GPU
    //std::cout << "1. ...\n";
    cudaEventRecord(start_h2d);
    
    float *device_data;
    cudaMalloc(&device_data, data_size * sizeof(float));
    cudaMemcpy(device_data, oriData.data(), data_size * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEventRecord(end_h2d);

    //2. CUDA GPU
    cudaEventRecord(start_compress);
    
    auto compressor = ndzip::make_cuda_compressor<float>(
        ndzip::compressor_requirements{data_extent});

    size_t compressed_size = ndzip::compressed_length_bound<float>(data_extent);
    ndzip::compressed_type<float> *device_compressed_data;
    cudaMalloc(&device_compressed_data, compressed_size * sizeof(ndzip::compressed_type<float>));

    ndzip::index_type *device_compressed_length;
    cudaMalloc(&device_compressed_length, sizeof(ndzip::index_type));

    //3. GPU
    //std::cout << "2. GPU ...\n";
    // cudaEventRecord(start_compress);
    
    compressor->compress(device_data, data_extent, device_compressed_data, device_compressed_length);
    
    cudaDeviceSynchronize(); //translated comment
    cudaEventRecord(end_compress);

    //translated comment
    //std::cout << "3. ...\n";
    cudaEventRecord(start_d2h_compress);
    
    ndzip::index_type host_compressed_length;
    cudaMemcpy(&host_compressed_length, device_compressed_length, sizeof(ndzip::index_type), cudaMemcpyDeviceToHost);

    std::vector<ndzip::compressed_type<float>> host_compressed_data(host_compressed_length);
    cudaMemcpy(host_compressed_data.data(), device_compressed_data,
               host_compressed_length * sizeof(ndzip::compressed_type<float>), cudaMemcpyDeviceToHost);
    
    cudaEventRecord(end_d2h_compress);

    //translated comment
    //std::cout << "\nCUDA \n";
    
    //translated comment
    //std::cout << "1. ...\n";
    cudaEventRecord(start_h2d_decompress);
    
    //translated comment
    cudaFree(device_data);
    cudaFree(device_compressed_data);
    
    //translated comment
    cudaMalloc(&device_compressed_data, host_compressed_length * sizeof(ndzip::compressed_type<float>));
    cudaMemcpy(device_compressed_data, host_compressed_data.data(),
               host_compressed_length * sizeof(ndzip::compressed_type<float>), cudaMemcpyHostToDevice);
    
    cudaEventRecord(end_h2d_decompress);

    //2. CUDA
    cudaEventRecord(start_decompress);
    auto decompressor = ndzip::make_cuda_decompressor<float>(1);
    
    float *device_decompressed_data;
    cudaMalloc(&device_decompressed_data, data_size * sizeof(float));

    //3. GPU
    //std::cout << "2. GPU ...\n";
    
    decompressor->decompress(device_compressed_data, device_decompressed_data, data_extent);
    
    cudaDeviceSynchronize(); //translated comment
    cudaEventRecord(end_decompress);

    //translated comment
    //std::cout << "3. ...\n";
    cudaEventRecord(start_d2h_decompress);
    
    std::vector<float> host_decompressed_data(data_size);
    cudaMemcpy(host_decompressed_data.data(), device_decompressed_data, data_size * sizeof(float),
               cudaMemcpyDeviceToHost);
    
    cudaEventRecord(end_d2h_decompress);
    cudaEventRecord(end_total);
    
    //translated comment
    cudaDeviceSynchronize();

    //translated comment
    float h2d_time, compress_time, d2h_compress_time;
    float h2d_decompress_time, decompress_time, d2h_decompress_time, total_time;
    
    cudaEventElapsedTime(&h2d_time, start_h2d, end_h2d);
    cudaEventElapsedTime(&compress_time, start_compress, end_compress);
    cudaEventElapsedTime(&d2h_compress_time, start_d2h_compress, end_d2h_compress);
    cudaEventElapsedTime(&h2d_decompress_time, start_h2d_decompress, end_h2d_decompress);
    cudaEventElapsedTime(&decompress_time, start_decompress, end_decompress);
    cudaEventElapsedTime(&d2h_decompress_time, start_d2h_decompress, end_d2h_decompress);
    cudaEventElapsedTime(&total_time, start_total, end_total);
    //translated comment
    float total_compress_time = h2d_time + compress_time + d2h_compress_time;
    float total_decompress_time = h2d_decompress_time + decompress_time + d2h_decompress_time;
    //translated comment
    double compress_throughput = (original_size / (1024.0 * 1024.0)) / (compress_time / 1000.0);
    double decompress_throughput = (original_size / (1024.0 * 1024.0)) / (decompress_time / 1000.0);
    double total_compress_throughput = (original_size / (1024.0 * 1024.0)) / (total_compress_time / 1000.0);
    double total_decompress_throughput = (original_size / (1024.0 * 1024.0)) / (total_decompress_time / 1000.0);

    //translated comment
    size_t compressed_size_out = host_compressed_length * sizeof(ndzip::compressed_type<float>);
    double compression_ratio = static_cast<double>(compressed_size_out) / original_size;
    CompressionInfo tmp{
        original_size/1024.0/1024.0,
        compressed_size_out/1024.0/1024.0,
        compression_ratio,
        compress_time,
        total_compress_time,
        total_compress_throughput/1024,
        decompress_time,
        total_decompress_time,
        total_decompress_throughput/1024,
    };
    ans=tmp;
    // std::cout << std::fixed << std::setprecision(3);
    //std::cout << " - GPU : " << decompress_time << " ms" << std::endl;
    //std::cout << " - : " << total_decompress_time << " ms" << std::endl;
    //std::cout << " - : " << total_compress_throughput << " MB/s (" << total_compress_throughput/1024 << " GB/s)" << std::endl;
    //std::cout << " - : " << total_decompress_throughput << " MB/s (" << total_decompress_throughput/1024 << " GB/s)" << std::endl;
    //std::cout << " - : " << compression_ratio << std::endl;

    //translated comment
    ASSERT_EQ(host_decompressed_data.size(), oriData.size()) << "解压失败，数据大小不一致";
    for (size_t i = 0; i < oriData.size(); ++i) {
        ASSERT_FLOAT_EQ(host_decompressed_data[i], oriData[i]) << "数据不一致，索引: " << i;
    }

    //GPU
    cudaFree(device_compressed_data);
    cudaFree(device_compressed_length);
    cudaFree(device_decompressed_data);

    //CUDA
    cudaEventDestroy(start_total);
    cudaEventDestroy(end_total);
    cudaEventDestroy(start_h2d);
    cudaEventDestroy(end_h2d);
    cudaEventDestroy(start_compress);
    cudaEventDestroy(end_compress);
    cudaEventDestroy(start_d2h_compress);
    cudaEventDestroy(end_d2h_compress);
    cudaEventDestroy(start_h2d_decompress);
    cudaEventDestroy(end_h2d_decompress);
    cudaEventDestroy(start_decompress);
    cudaEventDestroy(end_decompress);
    cudaEventDestroy(start_d2h_decompress);
    cudaEventDestroy(end_d2h_decompress);
    //std::cout << "CUDA 。\n";
}

//Google Test
TEST(NDZipCudaTest, CompressionDecompression) {
    //std::string dir_path = "../test/data/big"; //
    std::string dir_path = "../test/data/source"; //translated comment

    bool wram = 0;
    for (const auto &entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();
            if(!wram) {
                std::cout << "------------------warm---------------------------\n";
                wram = 1;
                test_compression(file_path);
                std::cout << "------------------warm_end---------------------------\n";
            }
            std::cout << "正在处理文件: " << file_path << "\n";
            test_compression(file_path);
            std::cout << "---------------------------------------------\n";
            cudaDeviceSynchronize();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
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
    else{
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }
}