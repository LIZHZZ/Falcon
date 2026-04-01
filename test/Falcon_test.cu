//translated comment

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
#define NUM_STREAMS 1 
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

CompressionInfo test_compression(ProcessedData data, size_t chunkSize)
{
    FalconPipeline ex(NUM_STREAMS);


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
        FalconPipeline ex(NUM_STREAMS);
        ex.executeCompressionPipeline(data, warmup_chunk);
        cudaDeviceSynchronize(); //translated comment
}

int setChunk(int nbEle)
{
    size_t chunkSize=1025;
    size_t temp=nbEle/NUM_STREAMS;// (data+temp-1)/temp<NUm_streams
    //translated comment
    // size_t availableMemory = getAvailableGPUMemory();
    size_t availableMemory, totalMem;
    cudaMemGetInfo(&availableMemory, &totalMem);
    // size_t limit=availableMemory/(4 * NUM_STREAMS * sizeof(double) * 2);
    size_t limit=64*1024*1024/sizeof(double);
    //16 chunkSize*NUM_STREAMS*8*sizeof(double) * 2<availableMemory/8*
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

//: CUDA
CompressionInfo comp_stream(std::vector<double> oriData, std::vector<double> &decompData)
{

    size_t nbEle = oriData.size();
    size_t cmpSize = 0; //translated comment

    //translated comment
    double* d_oriData;
    double* d_decData;
    unsigned char* d_cmpBytes;

    //CUDA
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //CUDA
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    
    cudaEvent_t g_startEvent, g_stopEvent;
    cudaEventCreate(&g_startEvent);
    cudaEventCreate(&g_stopEvent);
    
    //translated comment
    //std::cout << " GPU \n";
    cudaEventRecord(g_startEvent, stream);
    
    //translated comment
    //std::cout << "1. ...\n";
    cudaMalloc(&d_oriData, nbEle * sizeof(double));
    cudaMalloc(&d_cmpBytes, nbEle * sizeof(double));
    
    cudaMemcpyAsync(d_oriData, oriData.data(), nbEle * sizeof(double), cudaMemcpyHostToDevice, stream);

    //3. GPU
    //std::cout << "2. GPU ...\n";
    cudaEventRecord(startEvent, stream);
    FalconCompressor::Falcon_compress(d_oriData, d_cmpBytes, nbEle, &cmpSize, stream);
    
    cudaEventRecord(stopEvent, stream);
    //translated comment
    cudaStreamSynchronize(stream);


    //translated comment
    //std::cout << "3. ...\n";
    std::vector<unsigned char> h_cmpBytes(cmpSize);
    cudaMemcpyAsync(h_cmpBytes.data(), d_cmpBytes, cmpSize, cudaMemcpyDeviceToHost, stream);
    cudaEventRecord(g_stopEvent, stream);
    cudaStreamSynchronize(stream);
    float end_full_compress=0.0;
    float total_compress_time = 0.0;
    cudaEventElapsedTime(&end_full_compress, startEvent, stopEvent);
    cudaEventElapsedTime(&total_compress_time, g_startEvent,g_stopEvent);
    double compression_ratio = static_cast<double>(cmpSize) / (nbEle * sizeof(double));


    //translated comment
    cudaFree(d_oriData);
    cudaFree(d_cmpBytes);

    //translated comment
    cudaEvent_t startEvent1, stopEvent1;
    cudaEventCreate(&startEvent1);
    cudaEventCreate(&stopEvent1);
    
    cudaEvent_t g_startEvent1, g_stopEvent1;
    cudaEventCreate(&g_startEvent1);
    cudaEventCreate(&g_stopEvent1);

    cudaEventRecord(g_startEvent1, stream);
    
    //translated comment
    //std::cout << "1. ...\n";
    cudaMalloc(&d_cmpBytes, cmpSize);
        //translated comment
    cudaMalloc(&d_decData, nbEle * sizeof(double));
    cudaMemcpyAsync(d_cmpBytes, h_cmpBytes.data(), cmpSize, cudaMemcpyHostToDevice, stream);


    //3. GPU
    //std::cout << "2. GPU ...\n";
    cudaEventRecord(startEvent1, stream);
    FalconDecompressor GDFD;
    GDFD.Falcon_decompress_stream_optimized(d_decData, d_cmpBytes, nbEle, cmpSize, stream);
    cudaEventRecord(stopEvent1, stream);

    //translated comment
    //std::cout << "3. ...\n";
    decompData.resize(nbEle);
    cudaMemcpyAsync(decompData.data(), d_decData, nbEle * sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaEventRecord(g_stopEvent1, stream);
    cudaStreamSynchronize(stream);
    float end_full_decompress=0.0;
    float total_decompress_time = 0.0;
    cudaEventElapsedTime(&end_full_decompress, startEvent1, stopEvent1);
    cudaEventElapsedTime(&total_decompress_time, g_startEvent1,g_stopEvent1);
    size_t in_bytes = oriData.size() * sizeof(double);
    double data_size_gb = in_bytes / (1024.0 * 1024.0 * 1024.0);
    double compress_throughput = data_size_gb / (total_compress_time / 1000.0);
    double decompress_throughput = data_size_gb / (total_decompress_time / 1000.0);
    CompressionInfo tmp{
        in_bytes/1024.0/1024.0,
        static_cast<double>(cmpSize) /1024.0/1024.0,
        compression_ratio,
        end_full_compress,
        total_compress_time,
        compress_throughput,
        end_full_decompress,
        total_decompress_time,
        decompress_throughput};
    //translated comment
    cudaFree(d_cmpBytes);
    cudaFree(d_decData);
    cudaStreamDestroy(stream);

    //translated comment
    for (size_t i = 0; i < oriData.size(); ++i) {
        if(abs(oriData[i] -decompData[i])>1e-6) 
        {
            std::cout<< "第 " << i << " 个值不相等。\n";
            return tmp;
        }
    }
    printf("comp success\n");
    return tmp;
}
//translated comment
CompressionInfo test_stream_compression(const std::string& file_path) {
    //translated comment
    std::vector<double> oriData = read_data(file_path);
    std::vector<double> decompressedData;
    return comp_stream(oriData, decompressedData);

}

int main(int argc, char *argv[]) {
    cudaSetDevice(0);
    if (argc < 2) {
        printf("单流测试使用方法:\n");
        printf("  %s --file <file_path> : 从文件测试\n", argv[0]);
        printf("  %s --dir <directory_path> : 测试目录中所有文件\n", argv[0]);
        return 1;
    }

    std::string arg = argv[1];

    if (arg == "--file" && argc >= 3) {
        std::string file_path = argv[2];
        test(file_path);
    } else if (arg == "--old-dir" && argc >= 3) {
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
    else if (arg == "--dir" && argc >= 3) {//translated comment
        std::string dir_path = argv[2];
        //translated comment
        if (!fs::exists(dir_path)) {
            std::cerr << "指定的数据目录不存在: " << dir_path << std::endl;
            return 1;
        }
        bool warm=0;
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                std::string file_path = entry.path().string();
                CompressionInfo a;
                if(!warm)
                {
                    std::cout << "\n-------------------warm-------------------------- " << file_path << std::endl;
                    test_stream_compression(file_path);
                    warm=1;
                    std::cout << "-------------------warm_end------------------------" << std::endl;
                }
                std::cout <<"正在处理文件: " << file_path << std::endl;
                for(int i=0;i<3;i++)
                {
                    //GPU
                    cudaDeviceReset();
                    
                    a += test_stream_compression(file_path);
                    
                    //translated comment
                    cudaDeviceSynchronize(); 
                }
                a=a/3;
                a.print();
                std::cout << "---------------------------------------------" << std::endl;

            }
        }
    }
    else {
        printf("无效的参数. 使用 %s 查看用法\n", argv[0]);
        return 1;
    }

    return 0;
}

