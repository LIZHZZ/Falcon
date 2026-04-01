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
#include "Falcon_pipeline.cuh"
#include <thread>
namespace fs = std::filesystem;

std::string title = "";
int NUM_STREAM = 16; //16 CUDA Stream
CompressionInfo test_compression(ProcessedData data, size_t chunkSize);
CompressionInfo test_streams_compression(ProcessedData data, size_t chunkSize);
//translated comment
std::vector<double> generate_test_data(size_t nbEle, int pattern_type = 0) {
    std::vector<double> data(nbEle);

    std::random_device rd;
    std::mt19937 gen(rd());

    switch (pattern_type) {
        case 0: {
            //translated comment
            std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
            for (size_t i = 0; i < nbEle; ++i) {
                data[i] = dist(gen);
            }
            break;
        }
        case 1: {
            //translated comment
            for (size_t i = 0; i < nbEle; ++i) {
                data[i] = static_cast<double>(i) * 0.01;
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
                data[i] = static_cast<double>((i / step_size) * 100);
            }
            break;
        }
        default: {
            std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
            for (size_t i = 0; i < nbEle; ++i) {
                data[i] = dist(gen);
            }
        }
    }

    return data;
}



//translated comment
ProcessedData prepare_data(const std::string &source_path = "", size_t generate_size = 0, int pattern_type = 0) {
    ProcessedData result;
    std::vector<double> data;

    //translated comment
    if (generate_size > 0) {
        //translated comment
        //printf(" %zu ( : %d)\n", generate_size, pattern_type);
        data = generate_test_data(generate_size, pattern_type);
        result.nbEle = generate_size;
    } else if (!source_path.empty()) {
        //translated comment
        //printf(" : %s\n", source_path.c_str());
        data = read_data(source_path);
        result.nbEle = data.size();
    } else {
        printf("错误: 未指定数据源\n");
        result.nbEle = 0;
        result.oriData = nullptr;
        result.cmpBytes = nullptr;
        return result;
    }
    if(result.nbEle <= 0)
    {
        printf("wrong");
    }
    //translated comment
    cudaCheckError(cudaHostAlloc(&result.oriData, result.nbEle * sizeof(double), cudaHostAllocDefault));
    cudaCheckError(cudaHostAlloc((void**)&result.cmpBytes, result.nbEle * 1.2 * sizeof(double), cudaHostAllocDefault));
    cudaCheckError(cudaHostAlloc((void**)&result.cmpSize, sizeof(unsigned int), cudaHostAllocDefault));
    cudaCheckError(cudaHostAlloc(&result.decData, result.nbEle * sizeof(double), cudaHostAllocDefault));
    //translated comment
#pragma omp parallel for
    for (size_t i = 0; i < result.nbEle; ++i) {
        result.oriData[i] = data[i];
    }

    return result;
}

CompressionInfo test_compression(ProcessedData data, size_t chunkSize)
{

    FalconPipeline ex(NUM_STREAM);

    CompressionResult compResult = ex.executeCompressionPipelineNoBlock(data, chunkSize);
    cudaDeviceSynchronize(); 

    PipelineAnalysis decompAnalysis = ex.executeDecompressionPipeline(compResult, data);
    cudaDeviceSynchronize(); 


    return CompressionInfo{
        decompAnalysis.total_size,
        decompAnalysis.total_compressed_size,
        compResult.analysis.compression_ratio,
        0,
        compResult.analysis.comp_time,
        compResult.analysis.comp_throughout,
        0,
        decompAnalysis.decomp_time,
        decompAnalysis.decomp_throughout};
}

void warmup()
{
        size_t nbEle = (1*1024 * 1024) / sizeof(double);
        ProcessedData data = prepare_data("", nbEle, 0);
        if (data.nbEle == 0) {
            printf("错误: 无法读取文件数据\n");
            return ;
        }
        printf("GPU预热中...\n");
        size_t warmup_chunk = data.nbEle; //translated comment
        FalconPipeline ex;

        CompressionResult compResult = ex.executeCompressionPipelineSpare(data, warmup_chunk);
        cudaDeviceSynchronize(); //translated comment
}

int setChunk(int nbEle)
{
    size_t chunkSize=1025;
    size_t temp=nbEle/NUM_STREAM;// (data+temp-1)/temp<NUm_streams
    //translated comment
    // size_t availableMemory = getAvailableGPUMemory();
    size_t availableMemory, totalMem;
    cudaMemGetInfo(&availableMemory, &totalMem);
    // size_t limit=availableMemory/(4 * NUM_STREAM * sizeof(double) * 2);
    size_t limit=64*1025*1024/sizeof(double);

    //16 chunkSize*NUM_STREAMS*8*sizeof(double) * 2<availableMemory/8*
    while(chunkSize<=limit//MAX_NUMS_PER_CHUNK 
            && chunkSize<=temp)
    {
        chunkSize*=2;
    }
    // chunkSize=chunkSize/2;
    chunkSize=chunkSize>1025?chunkSize:1025;
    printf("chunkSize:%d\n",chunkSize);
    return chunkSize;
}

//translated comment
int test(const std::string &file_path = "", size_t data_size_mb = 0, int pattern_type = 0) {
    // warmup();
    cudaDeviceReset();

    if (!file_path.empty()) {
       
        std::vector<double> file_data = read_data(file_path);
        size_t nbEle = file_data.size();
        
        CompressionInfo a;
        for(int i = 0; i < 3; i++) {
            //GPU
            cudaDeviceReset();
            
            //CUDA
            ProcessedData data = prepare_data(file_path);
            
            //chunk
            size_t chunkSize = setChunk(data.nbEle);
            
            //translated comment
            auto tmp = test_compression(data, chunkSize);
            a += tmp;
            
            // printf("Iteration %d - a:%.6f, get:%.6f\n", i+1, a.compression_ratio/(i+1), tmp.compression_ratio);
            
            //translated comment
            cleanup_data(data);
        }
        a=a/3;
        a.print();
        // cleanup_data(data);
        return 0;
    } else if (data_size_mb > 0) {
        size_t nbEle = (data_size_mb * 1024 * 1024) / sizeof(double);
        
        // ProcessedData data = prepare_data("", nbEle, pattern_type);
        // chunkSize=setChunk(data.nbEle);
        // CompressionInfo a;
        // for(int i=0;i<3;i++)
        // {
        //     a+=test_compression(data,chunkSize);
        // }
        // a=a/3;
        // a.print();
        // cleanup_data(data);
        CompressionInfo a;
        for(int i = 0; i < 3; i++) {
            //GPU
            cudaDeviceReset();
            
            //CUDA
            ProcessedData data = prepare_data("", nbEle, pattern_type);
            
            //chunk
            size_t chunkSize = setChunk(data.nbEle);
            
            //translated comment
            auto tmp = test_compression(data, chunkSize);
            a += tmp;
            
            // printf("Iteration %d - a:%.6f, get:%.6f\n", i+1, a.compression_ratio/(i+1), tmp.compression_ratio);
            
            //translated comment
            cleanup_data(data);
        }
        a=a/3;
        a.print();
        return 0;
    } else {
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
        //printf(" %s --generate <size_in_mb> [pattern_type] : \n", argv[0]);
        //printf(" pattern_type: 0= , 1= , 2= , 3= \n");
        //printf(" %s --analyze-blocks <file_path> : \n", argv[0]);
        //printf(" %s --analyze-blocks-gen <size_in_mb> [pattern_type] : \n", argv[0]);
        return 1;
    }

    std::string arg = argv[1];

    if (arg == "--file" && argc >= 3) {
        std::string file_path = argv[2];
        test(file_path);
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
                test(file_path);
                std::cout << "---------------------------------------------" << std::endl;
            }
        }
    } else if (arg == "--analyze-streams-dir" && argc >= 3) {
        std::string dir_path = argv[2];

        //translated comment
        if (!fs::exists(dir_path)) {
            std::cerr << "指定的数据目录不存在: " << dir_path << std::endl;
            return 1;
        }

        int streams[5] = {1, 4, 8, 16, 32};

        for (const auto &entry: fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                std::string file_path = entry.path().string();
                std::cout << "正在处理文件: " << file_path << std::endl;
                for (int i = 0; i < 5; i++) {
                    NUM_STREAM = streams[i];
                    printf("=================================================\n");
                    printf("=====Testing Stream : %d muti streams ======\n", NUM_STREAM);
                    printf("=================================================\n");
                    CompressionInfo a;
                    for (int j = 0; j < 3; j++) {
                        //GPU
                        cudaDeviceReset();

                        //CUDA
                        ProcessedData data = prepare_data(file_path);

                        //chunk
                        size_t chunkSize = setChunk(data.nbEle);

                        //translated comment
                        auto tmp = test_compression(data, chunkSize);
                        a += tmp;

                        cleanup_data(data);
                    }
                    a = a / 3;
                    a.print();
                    std::cout << "---------------------------------------------" << std::endl;
                }
            }
        }
    // } else if (arg == "--generate" && argc >= 3) {
    //     size_t data_size_mb = std::stoul(argv[2]);
    //     int pattern_type = (argc >= 4) ? std::stoi(argv[3]) : 0;

    //     test("", data_size_mb, pattern_type);
    // } else if (arg == "--analyze-blocks" && argc >= 3) {
    //     std::string file_path = argv[2];
    //     title = "analyze-blocks " + file_path;
    //translated comment
    //// 16mbB 512MB，
    //     std::vector<size_t> block_sizes = generate_power2_blocksizes(16*1024/4, 2*64*1024);

    //     test_multiple_blocksizes(file_path, block_sizes);
    // } else if (arg == "--analyze-blocks-gen" && argc >= 3) {
    //     size_t data_size_mb = std::stoul(argv[2]);
    //     int pattern_type = (argc >= 4) ? std::stoi(argv[3]) : 0;

    //     std::string pattern_str = (argc >= 4) ? argv[3] : "0";
    //     title = std::string("analyze-blocks-gen ") + argv[2] + " " + pattern_str;

    //translated comment
    //// 16mbB 512MB，
    //     std::vector<size_t> block_sizes = generate_power2_blocksizes(16*1024, 8*64*1024);

    //     test_multiple_blocksizes_generated(data_size_mb, block_sizes, pattern_type);
    // } else if (arg == "--analyze-blocks-custom" && argc >= 3) {
    //     std::string file_path = argv[2];

    //translated comment
    //     std::vector<size_t> block_sizes;

    //translated comment
    //     for (int i = 3; i < argc; i++) {
    //         block_sizes.push_back(std::stoul(argv[i]));
    //     }

    //     if (block_sizes.empty()) {
    //std::cerr << " : " << std::endl;
    //         return 1;
    //     }

    //     test_multiple_blocksizes(file_path, block_sizes);
    } else {
        printf("无效的参数. 使用 %s 查看用法\n", argv[0]);
        return 1;
    }

    return 0;
}

