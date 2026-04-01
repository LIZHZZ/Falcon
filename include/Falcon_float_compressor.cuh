//
// Created by lz on 24-9-26.
//
// Falcon_float_compressor.cuh

#include <cuda_runtime.h>
#include <thread> 
#include <vector>
#include <cmath>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <thrust/device_vector.h>
// Compression constants
static const int cmp_tblock_size = 32; // Each thread block contains 32 constants. Fixed to 32, cannot be modified.
static const int dec_tblock_size = 32; // Fixed to 32, cannot be modified.
static const int cmp_chunk = 1025;     // Each thread block contains 1025 elements; one thread processes 1025/32 elements
static const int dec_chunk = 1025;

#define BLOCK_SIZE_G 32
#define POW_NUM_G ((1L << 22) + (1L << 23))
#define DATA_PER_THREAD 1025//
#define MAX_NUMS_PER_CHUNK 1025*1024*4
#define DATA_PER_ONE 32

#define MAX_BITCOUNT 32
#define MAX_BITSIZE_PER_BLOCK (32 + 32 + 8 + 8 + 8 + 32 + (DATA_PER_THREAD) * MAX_BITCOUNT)
#define MAX_BYTES_PER_BLOCK ((MAX_BITSIZE_PER_BLOCK + 7) / 8)

class FalconCompressor {
public:
    FalconCompressor(){}

    void compress(const std::vector<float>& input, std::vector<unsigned char>& output);
    static void Falcon_compress(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, cudaStream_t stream);
    static void Falcon_compress_stream(float* d_oriData, unsigned char* d_cmpBytes, unsigned int* d2h_async_totalBits_ptr, size_t nbEle, cudaStream_t stream);

private:

    void setupDeviceMemory(
    const std::vector<float>& input,
    float*& d_input,
    unsigned char*& d_output,
    uint64_t*& d_bitSizes
    ); 
    void freeDeviceMemory(
    float* d_input,
    unsigned char* d_output,
    uint64_t* d_bitSizes
    );

};
