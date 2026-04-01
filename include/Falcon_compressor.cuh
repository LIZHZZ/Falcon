//
// Falcon_compressor.cuh

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
#define POW_NUM_G ((1L << 51) + (1L << 52))
#define DATA_PER_THREAD 1025//
#define MAX_NUMS_PER_CHUNK 1025*1024*8
#define DATA_PER_ONE 32

#define MAX_BITCOUNT 64
#define MAX_BITSIZE_PER_BLOCK (64 + 64 + 8 + 8 +8 + 64 + (DATA_PER_THREAD) * MAX_BITCOUNT)
#define MAX_BYTES_PER_BLOCK ((MAX_BITSIZE_PER_BLOCK + 7) / 8)

class FalconCompressor {
public:
    FalconCompressor(){}

    void compress(const std::vector<double>& input, std::vector<unsigned char>& output);
    static void Falcon_compress(double* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, cudaStream_t stream);
    static void Falcon_compress_stream(double* d_oriData, unsigned char* d_cmpBytes, unsigned int* d2h_async_totalBits_ptr, size_t nbEle, cudaStream_t stream);
    //translated comment
    static void Falcon_compress_no_pack(double* d_oriData, unsigned char* d_cmpBytes, unsigned int* d2h_async_totalBits_ptr, size_t nbEle, cudaStream_t stream);
    //translated comment
    static void Falcon_compress_br(double* d_oriData, unsigned char* d_cmpBytes, unsigned int* d2h_async_totalBits_ptr, size_t nbEle, cudaStream_t stream);
    // spare
    static void Falcon_compress_spare(double* d_oriData, unsigned char* d_cmpBytes, unsigned int* d2h_async_totalBits_ptr, size_t nbEle, cudaStream_t stream);
    // // string
    // static void Falcon_compress_string(double* d_oriData, unsigned char* d_cmpBytes, unsigned int* d2h_async_totalBits_ptr, size_t nbEle, cudaStream_t stream);
private:

    void setupDeviceMemory(
    const std::vector<double>& input,
    double*& d_input,
    unsigned char*& d_output,
    uint64_t*& d_bitSizes
    ); 
    void freeDeviceMemory(
    double* d_input,
    unsigned char* d_output,
    uint64_t* d_bitSizes
    );

};


