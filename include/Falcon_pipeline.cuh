// #include "data/dataset_utils.hpp"
// #include "Falcon_decompressor.cuh"
// #include "Falcon_compressor.cuh"

// Falcon_pipeline.cuh
// Pipeline compression and decompression class
// Path: Falcon/include/Falcon_pipeline.cuh

#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <cstddef>
#include "Falcon_compressor.cuh"
#include "Falcon_decompressor.cuh"

// Pipeline performance analysis structure
struct PipelineAnalysis {
    float total_size = 0;           // Total data size (MB)
    float compression_ratio = 0;    // Compression ratio
    float comp_time = 0;            // Compression time (ms)
    float decomp_time = 0;          // Decompression time (ms)
    float comp_throughout = 0;      // Compression throughput (GB/s)
    float decomp_throughout = 0;    // Decompression throughput (GB/s)
    float total_compressed_size = 0; // Total compressed size (MB)
    size_t chunk_size = 0;          // Chunk size
};

// Compression result structure
struct CompressionResult {
    PipelineAnalysis analysis;
    std::vector<size_t> chunkSizes;        // Compressed size of each chunk (bytes)
    std::vector<size_t> chunkElementCounts; // Number of elements in each chunk
    size_t totalChunks;                    // Total number of chunks
};

// Compressed data structure
struct CompressedData {
    unsigned char* cmpBytes;                // Compressed data buffer
    size_t totalCompressedSize;             // Total compressed size
    size_t totalElements;                   // Total number of original elements
    std::vector<size_t> chunkSizes;         // Compressed size of each chunk (bytes)
    std::vector<size_t> chunkElementCounts; // Number of elements in each chunk
    size_t totalChunks;                     // Total number of chunks
};

// Processed data structure
struct ProcessedData {
    double *oriData;
    unsigned char *cmpBytes;
    unsigned int *cmpSize;
    double *decData;
    size_t nbEle;
};

// Pipeline stage enum
enum Stage { 
    IDLE,           // Idle
    SIZE_PENDING,   // Waiting for size information
    DATA_PENDING    // Waiting for data transfer
};


// Pipeline compression and decompression class
class FalconPipeline {
public:
    FalconPipeline()
    {
        NUM_STREAMS = 16;
    }
    FalconPipeline(int numStreams);
    ~FalconPipeline();

    // Execute compression pipeline
    CompressionResult executeCompressionPipeline(
        ProcessedData &data, 
        size_t chunkSize);

    // Execute compression pipeline with a specified number of streams
    CompressionResult executeCompressionPipeline(
        ProcessedData &data, 
        size_t chunkSize,
        int numStreams);

    // Execute decompression pipeline
    PipelineAnalysis executeDecompressionPipeline(
        const CompressionResult& compResult,
        ProcessedData &decompData,
        bool visualize = true);

    // Execute decompression pipeline with a specified number of streams
    PipelineAnalysis executeDecompressionPipeline(
        const CompressionResult& compResult,
        ProcessedData &decompData,
        int numStreams,
        bool visualize = true);

    // Helper: create a CompressedData structure from a CompressionResult
    static CompressedData createCompressedData(
        const CompressionResult& compResult,
        const ProcessedData& originalData);

    // Set the number of streams
    void setNumStreams(int numStreams) { 
        NUM_STREAMS = numStreams; 
    }

    // Get the current number of streams
    int getNumStreams() const { 
        return NUM_STREAMS; 
    }

// Ablation study 1: kernel functions

    // Ablation 1.1: fully sparse vs fully dense (decompression is identical to the standard version)
        //  NoPack

    CompressionResult executeCompressionPipelineNoPack(
    ProcessedData &data, 
    size_t chunkSize);

        //  Spare

    CompressionResult executeCompressionPipelineSpare(
    ProcessedData &data, 
    size_t chunkSize);

    // Ablation 1.2: brute-force computation
    
        //  Br
    
    CompressionResult executeCompressionPipelineBr(
    ProcessedData &data, 
    size_t chunkSize);   

// Ablation study 3: pipeline behavior

    // Blocking

    CompressionResult executeCompressionPipelineBlock(
        ProcessedData &data,
        size_t chunkSize);

    // Non-blocking

    CompressionResult executeCompressionPipelineNoBlock(
        ProcessedData &data,
        size_t chunkSize);

private:
    int NUM_STREAMS;  // Number of CUDA streams

    // Internal implementation functions
    CompressionResult executeCompressionPipelineImpl(
        ProcessedData &data,
        size_t chunkSize,
        int numStreams);

    PipelineAnalysis executeDecompressionPipelineImpl(
        const CompressionResult& compResult,
        ProcessedData &decompData,
        int numStreams,
        bool visualize);
    
// Ablation implementations (kernel functions)
    
    // NOPACK
    CompressionResult executeCompressionPipelineImpl_NoPack(
    ProcessedData &data,
    size_t chunkSize,
    int numStreams);


    // SPARE
    CompressionResult executeCompressionPipelineImpl_Spare(
    ProcessedData &data,
    size_t chunkSize,
    int numStreams);

    // BR
    CompressionResult executeCompressionPipelineImpl_Br(
    ProcessedData &data,
    size_t chunkSize,
    int numStreams);

};

// CUDA错误检查宏
#ifndef cudaCheckError
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#endif

// 清理资源函数
void cleanup_data(ProcessedData &data);

