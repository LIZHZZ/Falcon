#include "elf.h"
#include <gtest/gtest.h>
#include <vector>
#include <chrono>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstdint> //uint8_t
#include "data/dataset_utils.hpp"
#include <filesystem>
namespace fs = std::filesystem;
//translated comment
std::vector<float> generate_elf_test_data(size_t size, int pattern_type) {
    std::vector<float> data(size);
    const float amplitude = 20.0f;
    
    switch(pattern_type) {
        case 0: //translated comment
            for(size_t i=0; i<size; ++i) {
                data[i] = amplitude * (static_cast<float>(rand())/RAND_MAX - 0.5f);
            }
            break;
            
        case 1: //translated comment
            for(size_t i=0; i<size; ++i) {
                data[i] = 0.0001f + i*0.000001f;
            }
            break;
            
        case 2: //translated comment
            for(size_t i=0; i<size; ++i) {
                float base = amplitude * std::sin(i * 0.1f);
                data[i] = base + 0.1f*(static_cast<float>(rand())/RAND_MAX - 0.5f);
            }
            break;
    }
    return data;
}
/*
//translated comment
CompressionInfo test_elf_compression(const std::vector<double>& original, double error_bound) {
    //translated comment
    uint8_t* compressed = nullptr;
    auto compress_start = std::chrono::high_resolution_clock::now();
    
    ssize_t compressed_size = elf_encode(
        const_cast<double*>(original.data()), 
        original.size(),
        &compressed,
        error_bound
    );
    
    auto compress_end = std::chrono::high_resolution_clock::now();
    
    //translated comment
    std::vector<double> decompressed(original.size());
    auto decompress_start = std::chrono::high_resolution_clock::now();
    
    ssize_t dec_size = elf_decode(
        compressed,
        compressed_size,
        decompressed.data(),
        error_bound
    );
    
    auto decompress_end = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < original.size()  ; ++i) {
        if(original[i] - decompressed[i] != 0) {
          GTEST_LOG_(INFO) << " " << original[i] << " " << decompressed[i];
        }
      }
    //translated comment
    const double original_bytes = original.size() * sizeof(double);
    const double compress_ratio = compressed_size/original_bytes;
    const double compress_time = std::chrono::duration<double, std::milli>(compress_end - compress_start).count();
    const double decompress_time = std::chrono::duration<double, std::milli>(decompress_end - decompress_start).count();
    
    const double original_GB = original_bytes / (1024.0 * 1024.0 * 1024.0);

    //translated comment
    const double compress_sec = compress_time / 1000.0;
    const double decompress_sec = decompress_time / 1000.0;
    //（GB/s）
    const double compression_throughput_GBs = original_GB / compress_sec;
    const double decompression_throughput_GBs = original_GB / decompress_sec;
    
    free(compressed); //translated comment
    return CompressionInfo{
        original_bytes/1024.0/1024.0,
        compressed_size/1024.0/1024.0,
        compress_ratio,
        0,
        compress_time,
        compression_throughput_GBs,0,decompress_time,decompression_throughput_GBs};
}
*/
CompressionInfo test_elf_compression(const std::vector<float>& original, float error_bound, size_t blockSize=-1) {
    if (blockSize <= 0) {
        blockSize=original.size();
        //std::cerr << " ：blockSize 0" << std::endl;
        // return CompressionInfo{};
    }
    blockSize=blockSize>original.size()?original.size():blockSize;
    const size_t totalSize = original.size();
    const size_t numBlocks = (totalSize + blockSize - 1) / blockSize; //translated comment
    
    std::cout << "分块信息: 总数据量=" << totalSize << ", 块大小=" << blockSize 
              << ", 块数量=" << numBlocks << std::endl;
    
    //translated comment
    double total_original_bytes = 0;
    double total_compressed_bytes = 0;
    double total_compress_time = 0;
    double total_decompress_time = 0;
    size_t successful_blocks = 0;
    
    //translated comment
    std::vector<float> all_decompressed;
    all_decompressed.reserve(totalSize);
    
    for (size_t blockIdx = 0; blockIdx < numBlocks; ++blockIdx) {
        const size_t startIdx = blockIdx * blockSize;
        const size_t endIdx = std::min(startIdx + blockSize, totalSize);
        const size_t currentBlockSize = endIdx - startIdx;
        
        //std::cout << " " << (blockIdx + 1) << "/" << numBlocks
        //<< " ( : " << currentBlockSize << ")" << std::endl;
        
        //translated comment
        std::vector<float> blockData(original.begin() + startIdx, original.begin() + endIdx);
        
        //translated comment
        uint8_t* compressed = nullptr;
        auto compress_start = std::chrono::high_resolution_clock::now();
        
        ssize_t compressed_size = elf_encode_32(
            blockData.data(), 
            currentBlockSize,
            &compressed,
            error_bound
        );
        
        auto compress_end = std::chrono::high_resolution_clock::now();
        
        if (compressed_size <= 0) {
            std::cerr << "块 " << (blockIdx + 1) << " 压缩失败" << std::endl;
            if (compressed) free(compressed);
            continue;
        }
        
        //translated comment
        std::vector<float> blockDecompressed(currentBlockSize);
        auto decompress_start = std::chrono::high_resolution_clock::now();
        
        ssize_t dec_size = elf_decode_32(
            compressed,
            compressed_size,
            blockDecompressed.data(),
            error_bound
        );
        
        auto decompress_end = std::chrono::high_resolution_clock::now();
        
        if (dec_size != currentBlockSize) {
            std::cerr << "块 " << (blockIdx + 1) << " 解压失败，期望大小: " 
                      << currentBlockSize << ", 实际大小: " << dec_size << std::endl;
            free(compressed);
            continue;
        }
        
        //translated comment
        bool block_valid = true;
        for (size_t i = 0; i < currentBlockSize; ++i) {
            if (std::abs(blockData[i] - blockDecompressed[i]) > error_bound) {
                std::cerr << "块 " << (blockIdx + 1) << " 数据验证失败，位置 " << i 
                          << ": 原始=" << blockData[i] << ", 解压=" << blockDecompressed[i] << std::endl;
                block_valid = false;
                break;
            }
        }
        
        if (!block_valid) {
            free(compressed);
            continue;
        }
        
        //translated comment
        const double block_original_bytes = currentBlockSize * sizeof(float);
        const double block_compress_time = std::chrono::duration<double, std::milli>(compress_end - compress_start).count();
        const double block_decompress_time = std::chrono::duration<double, std::milli>(decompress_end - decompress_start).count();
        
        total_original_bytes += block_original_bytes;
        total_compressed_bytes += compressed_size;
        total_compress_time += block_compress_time;
        total_decompress_time += block_decompress_time;
        successful_blocks++;
        
        //translated comment
        all_decompressed.insert(all_decompressed.end(), blockDecompressed.begin(), blockDecompressed.end());
        
        //translated comment
        const double block_ratio = static_cast<float>(compressed_size) / block_original_bytes;
        //std::cout << " : " << block_ratio << ", : " << block_compress_time
        //<< "ms, : " << block_decompress_time << "ms" << std::endl;
        
        free(compressed);
    }
    
    //translated comment
    if (successful_blocks == 0) {
        std::cerr << "所有块处理失败" << std::endl;
        return CompressionInfo{};
    }
    
    const double total_compress_ratio = total_compressed_bytes / total_original_bytes;
    const double total_original_GB = total_original_bytes / (1024.0 * 1024.0 * 1024.0);
    const double total_compress_sec = total_compress_time / 1000.0;
    const double total_decompress_sec = total_decompress_time / 1000.0;
    const double compression_throughput_GBs = total_original_GB / total_compress_sec;
    const double decompression_throughput_GBs = total_original_GB / total_decompress_sec;
    
    //std::cout << "\n :" << std::endl;
    //std::cout << " : " << successful_blocks << "/" << numBlocks << std::endl;
    //std::cout << " : " << total_compress_ratio << std::endl;
    //std::cout << " : " << total_compress_time << "ms" << std::endl;
    //std::cout << " : " << total_decompress_time << "ms" << std::endl;
    //std::cout << " : " << compression_throughput_GBs << " GB/s" << std::endl;
    //std::cout << " : " << decompression_throughput_GBs << " GB/s" << std::endl;
    
    return CompressionInfo{
        total_original_bytes/1024.0/1024.0,
        total_compressed_bytes/1024.0/1024.0,
        total_compress_ratio,
        0,
        total_compress_time,
        compression_throughput_GBs,
        0,
        total_decompress_time,
        decompression_throughput_GBs
    };
}

//translated comment
TEST(ElfCompressorTest, SmallDataset) {
    const std::vector<float> data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    test_elf_compression(data, 0.01f);
}

TEST(ElfCompressorTest, IncrementalData) {
    auto data = generate_elf_test_data(1024*256, 1); //translated comment
    test_elf_compression(data, 0.005f);
}

TEST(ElfCompressorTest, PeriodicWithNoise) {
    auto data = generate_elf_test_data(1024*1024, 2); //translated comment
    test_elf_compression(data, 0.02f);
}

//translated comment
CompressionInfo test_elf_with_file(const std::string& file_path, float error_bound) {
    std::vector<float> data = read_data_float(file_path);
    return test_elf_compression(data, error_bound,1024*1024);
}

//translated comment
TEST(ElfCompressorTest, LocalDataset) {
    //const std::string data_dir = "../test/data/big"; //
    const std::string data_dir = "../test/data/float"; //translated comment

    const float error_bound = 0.000; //translated comment
    
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (entry.is_regular_file()) {
            std::cout << "\n正在处理文件:  " << entry.path().string() << std::endl;
            test_elf_with_file(entry.path().string(), error_bound);
        }
    }
}
int main(int argc, char *argv[]) {
    
    std::string arg = argv[1];
    
    if (arg == "--dir" && argc >= 3) {

        std::string dir_path = argv[2];

        //translated comment
        if (!fs::exists(dir_path)) {
            std::cerr << "指定的数据目录不存在: " << dir_path << std::endl;
            return 1;
        }
        
        //const double error_bound = 0.000; //
        
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                CompressionInfo ans;
                std::cout << "\n正在处理文件: " << entry.path().string() << std::endl;
                for(int i=0;i<3;i++)
                {
                    ans+=test_elf_with_file(entry.path().string(), 0);
                }
                ans=ans/3;
                ans.print();
                std::cout << "\n---------------------------------------------" << std::endl;
            }
        }
    }
    else{
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }
}