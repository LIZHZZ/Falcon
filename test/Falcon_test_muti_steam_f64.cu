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
int NUM_STREAM=16; //16 CUDA Stream
//#define NUM_STREAMS 16 // 16 CUDA Stream
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


CompressionInfo test_compression(ProcessedData data, size_t chunkSize);
//translated comment
ProcessedData prepare_data(const std::string &source_path = "", size_t generate_size = 0, int pattern_type = 0,int fig=-1) {
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
        data = read_data(source_path,fig);
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
    cudaCheckError(cudaHostAlloc((void**)&result.cmpBytes, result.nbEle * sizeof(double), cudaHostAllocDefault));
    cudaCheckError(cudaHostAlloc((void**)&result.cmpSize, sizeof(unsigned int), cudaHostAllocDefault));
    cudaCheckError(cudaHostAlloc(&result.decData, result.nbEle * sizeof(double), cudaHostAllocDefault));
    //translated comment
#pragma omp parallel for
    for (size_t i = 0; i < result.nbEle; ++i) {
        result.oriData[i] = data[i];
    }

    return result;
}


CompressionInfo test_streams_compression(ProcessedData data, size_t chunkSize)
{

    FalconPipeline ex(NUM_STREAM);

    CompressionResult compResult = ex.executeCompressionPipeline(data, chunkSize);
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


CompressionInfo test_compression(ProcessedData data, size_t chunkSize)
{
    FalconPipeline ex;


    CompressionResult compResult = ex.executeCompressionPipeline(data, chunkSize);
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

//translated comment
int test_multiple_blocksizes(const std::string &file_path, const std::vector<size_t> &block_sizes_kb)
{
    printf("=================================================\n");
    printf("=====Testing Block Sizes : %d muti streams ======\n", NUM_STREAM);
    printf("=================================================\n");

    //translated comment
    for (size_t block_size_kb : block_sizes_kb)
    {
        //KB (double = 8 bytes)
        cudaDeviceReset();

        size_t chunkSize = (block_size_kb * 1025) / sizeof(double);

        printf("\n[测试块大小: %zu MB (%zu 元素)]\n", block_size_kb / 1024, chunkSize);

        CompressionInfo a;
        for (int i = 0; i < 3; i++)
        {
            //GPU
            cudaDeviceReset();

            //CUDA
            ProcessedData data = prepare_data(file_path);

            //translated comment
            auto tmp = test_compression(data, chunkSize);
            cudaDeviceSynchronize();
            a += tmp;

            // printf("Iteration %d - a:%.6f, get:%.6f\n", i+1, a.compression_ratio/(i+1), tmp.compression_ratio);

            //translated comment
            cleanup_data(data);
            // cudaFree(0);
        }
        a = a / 3;
        a.print();

    }

    return 0;
}

//translated comment
int test_multiple_blocksizes_generated(size_t data_size_mb, const std::vector<size_t> &block_sizes_kb,
                                       int pattern_type = 0) {
    printf("=================================================\n");
    printf("=======Testing FALCON with Different Block Sizes===\n");
    printf("=================================================\n");


    //translated comment
    //MB ( double ，8 )
    size_t nbEle = (data_size_mb * 1024 * 1024) / sizeof(double);
    ProcessedData data = prepare_data("", nbEle, pattern_type);
    if (data.nbEle == 0) {
        printf("错误: 无法读取文件数据\n");
        return 1;
    }
    printf("GPU预热中...\n");
    size_t warmup_chunk = data.nbEle; //translated comment
    FalconPipeline ex;

    ex.executeCompressionPipeline(data, warmup_chunk);
    cudaDeviceSynchronize(); //translated comment


    //GPU
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    // size_t poolSize = freeMem * 0.4;
    //poolSize = (poolSize + 1024 * 2 * sizeof(double) - 1) & ~(1024 * 2 * sizeof(double) - 1); //

    std::vector<PipelineAnalysis> results;

    //translated comment
    for (size_t block_size_kb: block_sizes_kb) {
        //KB (double = 8 bytes)
        size_t chunkSize = (block_size_kb * 1024) / sizeof(double);

        printf("\n[测试块大小: %zu KB (%zu 元素)]\n", block_size_kb, chunkSize);

        //translated comment
        for(int i=0;i<3;i++)
        {
            FalconPipeline ex;
            CompressionResult compResult = ex.executeCompressionPipeline(data, chunkSize);
            PipelineAnalysis result = compResult.analysis;
            results.push_back(result);
        }

        //， GPU
        std::this_thread::sleep_for(std::chrono::milliseconds(500*3));
    }

    //translated comment
    // visualize_stage_timing_relationship(results);

    //CSV
    // output_blocksize_timing_csv(results, "block_size_timing_analysis_generated.csv");

    //translated comment
    cleanup_data(data);

    return 0;
}

//translated comment
std::vector<size_t> generate_power2_blocksizes(size_t min_kb, size_t max_kb) {
    std::vector<size_t> sizes;
    for (size_t size = min_kb; size <= max_kb; size *= 2) {
        sizes.push_back(size);
    }
    return sizes;
}

//translated comment
std::vector<size_t> generate_linear_blocksizes(size_t min_kb, size_t max_kb, size_t step_kb) {
    std::vector<size_t> sizes;
    for (size_t size = min_kb; size <= max_kb; size += step_kb) {
        sizes.push_back(size);
    }
    return sizes;
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
        ex.executeCompressionPipeline(data, warmup_chunk);
        cudaDeviceSynchronize(); //translated comment
}

int setChunk(int nbEle)
{
    size_t chunkSize=1025;
    size_t temp=nbEle/NUM_STREAM;// (data+temp-1)/temp<NUm_stream
    //translated comment
    // size_t availableMemory = getAvailableGPUMemory();
    size_t availableMemory, totalMem;
    cudaMemGetInfo(&availableMemory, &totalMem);
    // size_t limit=availableMemory/(4 * NUM_STREAM * sizeof(double) * 2);
    size_t limit=64*1025*1024/sizeof(double);
    //16 chunkSize*NUM_STREAM*8*sizeof(double) * 2<availableMemory/8*
    while(chunkSize<=limit//MAX_NUMS_PER_CHUNK l
            && chunkSize<=temp)
    {
        chunkSize*=2;
    }
    chunkSize=chunkSize>limit?chunkSize/2:chunkSize;
    printf("chunkSize:%zu MB\n",chunkSize*sizeof(double)/1024/1025);
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
        printf("  %s --generate <size_in_mb> [pattern_type] : 生成数据测试\n", argv[0]);
        printf("    pattern_type: 0=随机数据, 1=线性增长, 2=正弦波, 3=阶梯\n");
        printf("  %s --analyze-blocks <file_path> : 分析不同块大小的性能\n", argv[0]);
        printf("  %s --analyze-blocks-gen <size_in_mb> [pattern_type] : 使用生成数据分析不同块大小的性能\n", argv[0]);
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
        warmup();
        for (const auto &entry: fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                std::string file_path = entry.path().string();
                std::cout << "正在处理文件: " << file_path << std::endl;
                test(file_path);
                std::cout << "---------------------------------------------" << std::endl;
            }
        }
    }
    else if(arg == "--file-beta" && argc >= 3){
        std::string file_path = argv[2];
        warmup();
        for(int beta=4;beta<18;beta++)
        {
            std::cout << "\n正在处理文件: " << file_path << " beta :" << beta << std::endl;
            
            CompressionInfo a;
            for(int i = 0; i < 1; i++) {
                //GPU
                cudaDeviceReset();
                
                //CUDA
                ProcessedData data = prepare_data(file_path,0,0,beta);
                
                //chunk
                size_t chunkSize = setChunk(data.nbEle);
                
                //translated comment
                auto tmp = test_compression(data, chunkSize);
                a += tmp;
                
                // printf("Iteration %d - a:%.6f, get:%.6f\n", i+1, a.compression_ratio/(i+1), tmp.compression_ratio);
                
                //translated comment
                cleanup_data(data);
            }
            // a=a/3;
            a.print();
            std::cout << "---------------------------------------------" << std::endl;

        }

        // cleanup_data(data);

    } 
    else if (arg == "--generate" && argc >= 3) {
        size_t data_size_mb = std::stoul(argv[2]);
        int pattern_type = (argc >= 4) ? std::stoi(argv[3]) : 0;

        test("", data_size_mb, pattern_type);
    } else if (arg == "--analyze-blocks" && argc >= 3) {
        std::string file_path = argv[2];
        title = "analyze-blocks " + file_path;
        //translated comment
        //16mbB 512MB，
        std::vector<size_t> block_sizes = generate_power2_blocksizes(16*1024/4, 2*64*1024);

        test_multiple_blocksizes(file_path, block_sizes);
    } else if (arg == "--analyze-blocks-gen" && argc >= 3) {
        size_t data_size_mb = std::stoul(argv[2]);
        int pattern_type = (argc >= 4) ? std::stoi(argv[3]) : 0;

        std::string pattern_str = (argc >= 4) ? argv[3] : "0";
        title = std::string("analyze-blocks-gen ") + argv[2] + " " + pattern_str;

        //translated comment
        //16mbB 512MB，
        std::vector<size_t> block_sizes = generate_power2_blocksizes(16*1024, 8*64*1024);

        test_multiple_blocksizes_generated(data_size_mb, block_sizes, pattern_type);
    }else if (arg == "--analyze-blocks-dir" && argc >= 3)
    {
        std::string dir_path = argv[2];

        //translated comment
        if (!fs::exists(dir_path))
        {
            std::cerr << "指定的数据目录不存在: " << dir_path << std::endl;
            return 1;
        }
        //16mbB 512MB，
        std::vector<size_t> block_sizes = generate_power2_blocksizes(4*1024, 64*1024);
        // std::vector<size_t> block_sizes = {32 * 1024};
        // int streams[5]={1,4,8,16,32};
        for (const auto &entry : fs::directory_iterator(dir_path))
        {
            if (entry.is_regular_file())
            {
                std::string file_path = entry.path().string();
                std::cout << "正在处理文件: " << file_path << std::endl;
                // for(int i=0;i<5;i++)
                // {
                //     NUM_STREAM=streams[i];
                    test_multiple_blocksizes(file_path, block_sizes);
                    std::cout << "---------------------------------------------" << std::endl;
                // }
            }
        }
    } 
    else if(arg == "--analyze-streams-dir" && argc >= 3){
        std::string dir_path = argv[2];

        //translated comment
        if (!fs::exists(dir_path))
        {
            std::cerr << "指定的数据目录不存在: " << dir_path << std::endl;
            return 1;
        }

        int streams[5]={1,4,8,16,32};


        for (const auto &entry : fs::directory_iterator(dir_path))
        {
            if (entry.is_regular_file())
            {
                std::string file_path = entry.path().string();
                std::cout << "正在处理文件: " << file_path << std::endl;
                for(int i=0;i<5;i++)
                {
                    NUM_STREAM=streams[i];
                    printf("=================================================\n");
                    printf("=====Testing Stream : %d muti streams ======\n", NUM_STREAM);
                    printf("=================================================\n");
                    CompressionInfo a;
                    for(int i = 0; i < 3; i++) {
                        //GPU
                        cudaDeviceReset();
                        
                        //CUDA
                        ProcessedData data = prepare_data(file_path);
                        
                        //chunk
                        size_t chunkSize = setChunk(data.nbEle);
                        
                        //translated comment
                        auto tmp = test_streams_compression(data, chunkSize);
                        a += tmp;

                        cleanup_data(data);
                    }
                    a=a/3;
                    a.print();
                    std::cout << "---------------------------------------------" << std::endl;
                }
            }
        }

    }
    else {
        printf("无效的参数. 使用 %s 查看用法\n", argv[0]);
        return 1;
    }

    return 0;
}

