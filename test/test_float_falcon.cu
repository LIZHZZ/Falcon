#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include "data/dataset_utils.hpp"
#include "Falcon_float_pipeline.cuh"
#include <thread>
namespace fs = std::filesystem;

std::string title = "";
#define NUM_STREAMS 16 //16 CUDA Stream

//translated comment
std::vector<float> generate_test_data(size_t nbEle, int pattern_type = 0) {
    std::vector<float> data(nbEle);

    std::random_device rd;
    std::mt19937 gen(rd());

    switch (pattern_type) {
        case 0: {
            //translated comment
            std::uniform_real_distribution<float> dist(-1000.0, 1000.0);
            for (size_t i = 0; i < nbEle; ++i) {
                data[i] = dist(gen);
            }
            break;
        }
        case 1: {
            //translated comment
            for (size_t i = 0; i < nbEle; ++i) {
                data[i] = static_cast<float>(i) * 0.01;
            }
            break;
        }
        case 2: {
            //translated comment
            for (size_t i = 0; i < nbEle; ++i) {
                data[i] = 1000.0 * sin(0.01 * i);
            }
            break;
        }
        case 3: {
            //translated comment
            int step_size = nbEle / 10;
            for (size_t i = 0; i < nbEle; ++i) {
                data[i] = static_cast<float>((i / step_size) * 100);
            }
            break;
        }
        default: {
            std::uniform_real_distribution<float> dist(-1000.0, 1000.0);
            for (size_t i = 0; i < nbEle; ++i) {
                data[i] = dist(gen);
            }
        }
    }

    return data;
}


//translated comment
ProcessedData_32 prepare_data_32(const std::string &source_path = "", 
                                  size_t generate_size = 0, 
                                  int pattern_type = 0) {
    ProcessedData_32 result;
    std::vector<float> data;

    //translated comment
    if (generate_size > 0) {
        data = generate_test_data(generate_size, pattern_type);
        result.nbEle = generate_size;
    } else if (!source_path.empty()) {
        data = read_data_float(source_path);
        result.nbEle = data.size();
    } else {
        printf("错误: 未指定数据源\n");
        result.nbEle = 0;
        result.oriData = nullptr;
        result.cmpBytes = nullptr;
        return result;
    }
    
    if(result.nbEle <= 0) {
        printf("wrong");
    }
    
    //translated comment
    cudaCheckError_32(cudaHostAlloc(&result.oriData, 
        result.nbEle * sizeof(float), cudaHostAllocDefault));
    cudaCheckError_32(cudaHostAlloc((void**)&result.cmpBytes, 
        result.nbEle * sizeof(float), cudaHostAllocDefault));
    cudaCheckError_32(cudaHostAlloc((void**)&result.cmpSize, 
        sizeof(unsigned int), cudaHostAllocDefault));
    cudaCheckError_32(cudaHostAlloc(&result.decData, 
        result.nbEle * sizeof(float), cudaHostAllocDefault));
    
    //translated comment
    #pragma omp parallel for
    for (size_t i = 0; i < result.nbEle; ++i) {
        result.oriData[i] = data[i];
    }

    return result;
}
CompressionInfo test_compression(ProcessedData_32 data, size_t chunkSize)
{
    //translated comment
    FalconPipeline_32 pipeline;

    //translated comment
    CompressionResult_32 compResult = pipeline.executeCompressionPipeline(data, chunkSize);
    cudaDeviceSynchronize(); 

    //translated comment
    PipelineAnalysis_32 decompAnalysis = pipeline.executeDecompressionPipeline(compResult, data);
    cudaDeviceSynchronize(); 

    //translated comment
    return CompressionInfo{
        decompAnalysis.total_size,
        decompAnalysis.total_compressed_size,
        compResult.analysis.compression_ratio,
        0,
        compResult.analysis.comp_time,
        compResult.analysis.comp_throughout,
        0,
        decompAnalysis.decomp_time,
        decompAnalysis.decomp_throughout
    };
}

int setChunk(int nbEle)
{
    size_t chunkSize=1025;
    size_t temp=nbEle/NUM_STREAMS;// (data+temp-1)/temp<NUm_streams
    //translated comment
    // size_t availableMemory = getAvailableGPUMemory();
    size_t availableMemory, totalMem;
    cudaMemGetInfo(&availableMemory, &totalMem);
    size_t limit=availableMemory/(4 * NUM_STREAMS * sizeof(float) * 2);
    //16 chunkSize*NUM_STREAMS*8*sizeof(double) * 2<availableMemory/8*
    while(chunkSize<=limit//MAX_NUMS_PER_CHUNK 
            && chunkSize<=temp)
    {
        chunkSize*=2;
    }
    chunkSize=chunkSize/2;
    chunkSize=chunkSize>1024?chunkSize:1024;
    printf("chunkSize:%d\n",chunkSize);
    return chunkSize;
}
//test
int test_32(const std::string &file_path = "", size_t data_size_mb = 0, int pattern_type = 0) {
    cudaDeviceReset();

    if (!file_path.empty()) {
        std::vector<float> file_data = read_data_float(file_path);
        size_t nbEle = file_data.size();
        
        CompressionInfo a;
        for(int i = 0; i < 3; i++) {
            cudaDeviceReset();
            
            //translated comment
            ProcessedData_32 data = prepare_data_32(file_path);
            size_t chunkSize = setChunk(data.nbEle);
            
            //translated comment
            auto tmp = test_compression(data, chunkSize);
            a += tmp;
            
            cleanup_data_32(data);
        }
        a = a / 3;
        a.print();
        return 0;
    } 
    else if (data_size_mb > 0) {
        size_t nbEle = (data_size_mb * 1024 * 1024) / sizeof(float);
        
        CompressionInfo a;
        for(int i = 0; i < 3; i++) {
            cudaDeviceReset();
            
            ProcessedData_32 data = prepare_data_32("", nbEle, pattern_type);
            size_t chunkSize = setChunk(data.nbEle);
            
            auto tmp = test_compression(data, chunkSize);
            a += tmp;
            
            cleanup_data_32(data);
        }
        a = a / 3;
        a.print();
        return 0;
    } 
    else {
        printf("错误: 必须提供文件路径或数据生成参数\n");
        return 1;
    }
}
int main(int argc, char *argv[]) {
    cudaSetDevice(0);
    if (argc < 2) {
        printf("使用方法:\n");
        printf("  %s --file <file_path> : 从文件测试\n", argv[0]);
        printf("  %s --dir <directory_path> : 测试目录中所有文件\n", argv[0]);
        printf("  %s --generate <size_in_mb> [pattern_type] : 生成数据测试\n", argv[0]);
        printf("    pattern_type: 0=随机数据, 1=线性增长, 2=正弦波, 3=阶梯\n");
        return 1;
    }

    std::string arg = argv[1];

    if (arg == "--file" && argc >= 3) {
        std::string file_path = argv[2];
        test_32(file_path);
    } else if (arg == "--dir" && argc >= 3) {
        std::string dir_path = argv[2];

        //translated comment
        if (!fs::exists(dir_path)) {
            std::cerr << "指定的数据目录不存在: " << dir_path << std::endl;
            return 1;
        }

        for (const auto &entry: fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                std::string file_path = entry.path().string();
                std::cout << "正在处理文件: " << file_path << std::endl;
                test_32(file_path);
                std::cout << "---------------------------------------------" << std::endl;
            }
        }
    } else if (arg == "--generate" && argc >= 3) {
        size_t data_size_mb = std::stoul(argv[2]);
        int pattern_type = (argc >= 4) ? std::stoi(argv[3]) : 0;

        test_32("", data_size_mb, pattern_type);
    } else {
        printf("无效的参数. 使用 %s 查看用法\n", argv[0]);
        return 1;
    }

    return 0;
}
