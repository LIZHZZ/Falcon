// Falcon_float_pipeline.cuh
// 32-bit floating-point pipeline compression and decompression class
// Path: Falcon/include/Falcon_float_pipeline.cuh

#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <cstddef>
#include "Falcon_float_compressor.cuh"
#include "Falcon_float_decompressor.cuh"

// Pipeline performance analysis structure
struct PipelineAnalysis_32 {
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
struct CompressionResult_32 {
    PipelineAnalysis_32 analysis;
    std::vector<size_t> chunkSizes;        // Compressed size of each chunk (bytes)
    std::vector<size_t> chunkElementCounts; // Number of elements in each chunk
    size_t totalChunks;                    // Total number of chunks
};

// Compressed data structure
struct CompressedData_32 {
    unsigned char* cmpBytes;                // Compressed data buffer
    size_t totalCompressedSize;             // Total compressed size
    size_t totalElements;                   // Total number of original elements
    std::vector<size_t> chunkSizes;         // Compressed size of each chunk (bytes)
    std::vector<size_t> chunkElementCounts; // Number of elements in each chunk
    size_t totalChunks;                     // Total number of chunks
};

// Processed data structure
struct ProcessedData_32 {
    float *oriData;
    unsigned char *cmpBytes;
    unsigned int *cmpSize;
    float *decData;
    size_t nbEle;
};

// Pipeline stage enum
enum Stage_32 { 
    IDLE_32,           // Idle
    SIZE_PENDING_32,   // Waiting for size information
    DATA_PENDING_32    // Waiting for data transfer
};

// Pipeline compression and decompression class
class FalconPipeline_32 {
public:
    FalconPipeline_32() {
        NUM_STREAMS = 16;
    }
    
    FalconPipeline_32(int numStreams) : NUM_STREAMS(numStreams) {}
    ~FalconPipeline_32();

    // Execute compression pipeline
    CompressionResult_32 executeCompressionPipeline(
        ProcessedData_32 &data, 
        size_t chunkSize);

    // Execute compression pipeline with a specified number of streams
    CompressionResult_32 executeCompressionPipeline(
        ProcessedData_32 &data, 
        size_t chunkSize,
        int numStreams);

    // Execute decompression pipeline
    PipelineAnalysis_32 executeDecompressionPipeline(
        const CompressionResult_32& compResult,
        ProcessedData_32 &decompData,
        bool visualize = true);

    // Execute decompression pipeline with a specified number of streams
    PipelineAnalysis_32 executeDecompressionPipeline(
        const CompressionResult_32& compResult,
        ProcessedData_32 &decompData,
        int numStreams,
        bool visualize = true);

    // Helper: create a CompressedData structure from a CompressionResult
    static CompressedData_32 createCompressedData(
        const CompressionResult_32& compResult,
        const ProcessedData_32& originalData);

    // Set the number of streams
    void setNumStreams(int numStreams) { 
        NUM_STREAMS = numStreams; 
    }

    // Get the current number of streams
    int getNumStreams() const { 
        return NUM_STREAMS; 
    }

private:
    int NUM_STREAMS;  // Number of CUDA streams

    // Internal implementation functions
    CompressionResult_32 executeCompressionPipelineImpl(
        ProcessedData_32 &data,
        size_t chunkSize,
        int numStreams);

    PipelineAnalysis_32 executeDecompressionPipelineImpl(
        const CompressionResult_32& compResult,
        ProcessedData_32 &decompData,
        int numStreams,
        bool visualize);
};

// CUDA错误检查宏
#ifndef cudaCheckError_32
#define cudaCheckError_32(ans) { gpuAssert_32((ans), __FILE__, __LINE__); }
inline void gpuAssert_32(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#endif

// 清理资源函数
void cleanup_data_32(ProcessedData_32 &data);