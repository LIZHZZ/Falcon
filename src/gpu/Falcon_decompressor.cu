
//Falcon_decompressor.cu
//translated comment
//

#include "Falcon_decompressor.cuh"
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>

//translated comment
__constant__ double POW10_TABLE[16] = {
    1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0,
    100000000.0, 1000000000.0, 10000000000.0, 100000000000.0, 
    1000000000000.0, 10000000000000.0, 100000000000000.0, 1000000000000000.0
};

//ZigZag -
__device__ __forceinline__ int64_t zigzag_decode(uint64_t n) {
    return (int64_t)(n >> 1) ^ -((int64_t)(n & 1));
}

//translated comment
__device__ __forceinline__ uint64_t readBitsDevice(const unsigned char* buffer, size_t& bitPos, int n) {
    if (n == 0) return 0;
    if (n > 64) n = 64; //translated comment
    
    uint64_t result = 0;
    size_t startByte = bitPos / 8;
    int startBit = bitPos % 8;
    
    //translated comment
    if (startBit == 0 && n >= 8) {
        //translated comment
        int fullBytes = n / 8;
        for (int i = 0; i < fullBytes; i++) {
            result |= ((uint64_t)buffer[startByte + i]) << (i * 8);
        }
        int remainingBits = n % 8;
        if (remainingBits > 0) {
            uint64_t lastBits = buffer[startByte + fullBytes] & ((1 << remainingBits) - 1);
            result |= lastBits << (fullBytes * 8);
        }
    } else {
        //translated comment
        for (int i = 0; i < n; i++) {
            size_t byteIdx = (bitPos + i) / 8;
            int bitIdx = (bitPos + i) % 8;
            uint64_t bit = (buffer[byteIdx] >> bitIdx) & 1ULL;
            result |= (bit << i);
        }
    }
    
    bitPos += n;
    return result;
}

__device__ double decodeDoubleWithSignLast(uint64_t value) {
    uint64_t original = (value >> 1) ^ -((int64_t)(value & 1));
    union {
        uint64_t u;
        double d;
    } val;

    val.u = original;
    return val.d;
}

//translated comment
__global__ void decompressKernelOptimized(
    const unsigned char* __restrict__ compressedData,
    double* __restrict__ output,
    const int* __restrict__ offsets,
    // int numBlocks,
    int numDatas
) {
    int blockId = blockIdx.x * blockDim.x + threadIdx.x;
    // if (blockId >= numBlocks) return;
    int numData = min(numDatas - blockId * 1025, 1025);
    
    if (numData <= 0) return;
    //translated comment
    const unsigned char* blockData = compressedData + offsets[blockId];
    size_t bitPos = 0;


    //translated comment
    uint64_t bitSize = readBitsDevice(blockData, bitPos, 64);
    int64_t firstValue = (int64_t)readBitsDevice(blockData, bitPos, 64);
    unsigned char maxDecimalPlaces = (unsigned char)readBitsDevice(blockData, bitPos, 8);
    unsigned char maxBeta = (unsigned char)readBitsDevice(blockData, bitPos, 8);
    unsigned char bitCount = (unsigned char)readBitsDevice(blockData, bitPos, 8);

    if (bitCount == 0 || bitCount > 64) {
        //translated comment
        for (int i = 0; i < numData; i++) {
            output[blockId * 1025 + i] = 0.0;
        }
        return;
    }

    uint64_t flag1 = readBitsDevice(blockData, bitPos, 64);
    int dataByte = (numData-1 + 7) / 8;
    int flag2Size = (dataByte + 7) / 8;
    
    //translated comment
    uint8_t result[64][128];
    uint8_t flag2[64][128];
    
    //translated comment
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        #pragma unroll 4
        for (int j = 0; j < 128; j += 4) {
            *((uint32_t*)&flag2[i][j]) = 0;
        }
    }

    //translated comment
    for (int i = 0; i < bitCount; i++) {
        bool isSparse = (flag1 & (1ULL << i)) != 0;
        
        if (isSparse) {
            //translated comment
            for (int z = 0; z < flag2Size * 8; z++) {
                flag2[i][z] = (uint8_t)readBitsDevice(blockData, bitPos, 1);
            }
            
            //translated comment
            for (int j = 0; j < dataByte; j++) {
                if (flag2[i][j] != 0) {
                    result[i][j] = (uint8_t)readBitsDevice(blockData, bitPos, 8);
                } else {
                    result[i][j] = 0;
                }
            }
        } else {
            //translated comment
            for (int j = 0; j < dataByte; j++) {
                result[i][j] = (uint8_t)readBitsDevice(blockData, bitPos, 8);
            }
        }
    }
    // 11111
    //delta -
    uint64_t deltasZigzag[1024];
    #pragma unroll 4
    for (int j = 0; j < numData-1; j++) {
        uint64_t delta = 0;
        int byteIndex = j / 8;
        int bitIndex = 7 - (j % 8); //translated comment
        
        for (int i = 0; i < bitCount; i++) {
            uint8_t bitValue = (result[i][byteIndex] >> bitIndex) & 1;
            delta |= ((uint64_t)bitValue << (bitCount - 1 - i));
        }
        deltasZigzag[j] = delta;
    }

    //translated comment
    int64_t prevValue = firstValue;
    double scale = maxDecimalPlaces < 16 ? POW10_TABLE[maxDecimalPlaces] : pow(10.0, maxDecimalPlaces);
    bool useDirectConversion = (maxBeta > 15);

    if (useDirectConversion) {
        uint64_t bits = (uint64_t)prevValue;
        output[blockId * 1025] = decodeDoubleWithSignLast(bits);
    } else {
        output[blockId * 1025] = (double)prevValue/ scale;
    }
    // output[blockId * 1024] = useDirectConversion ? 
    //     *reinterpret_cast<double*>(&firstValue) : 
    //     (double)firstValue / scale;
    
    for (int i = 1; i < numData; i++) {
        int64_t delta = zigzag_decode(deltasZigzag[i-1]);
        prevValue += delta;
        
        if (useDirectConversion) {
            uint64_t bits = (uint64_t)prevValue;
            output[blockId * 1025 + i] = decodeDoubleWithSignLast(bits);
        } else {
            output[blockId * 1025 + i] = (double)prevValue / scale;
        }
    }
}

//translated comment
__global__ void calculateOffsetsKernel(const unsigned char* d_cmpBytes,
                                      int* d_offsets,
                                      int* d_numBlocks,
                                      size_t cmpSize,
                                      size_t nbEle) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return; //translated comment

    size_t bitPos = 0;
    size_t totalBits = cmpSize * 8;
    size_t minHeaderSize = 216;//192; // 64 + 64 + 8 + 8 + 64 bits
    int offsetCount = 0;
    int maxBlocks = (nbEle + 1024) / 1025;

    while (bitPos + minHeaderSize <= totalBits && offsetCount < maxBlocks) {
        d_offsets[offsetCount] = (bitPos+7) / 8;

        uint64_t bitSize = readBitsDevice(d_cmpBytes, bitPos, 64);

        if (bitSize < 64) break;
        //printf(" Chunk %d: offset=%d, size= %dbytes\n",offsetCount,d_offsets[offsetCount],(bitSize+7)/8);
        size_t nextPos = bitPos + bitSize - 64;
        if (nextPos > totalBits) break;

        bitPos = nextPos;
        offsetCount++;
    }

    *d_numBlocks = offsetCount;
}
//translated comment
void FalconDecompressor::Falcon_decompress_stream_optimized(double* d_decData,
                                     unsigned char* d_cmpBytes,
                                     size_t nbEle,
                                     size_t cmpSize,
                                     cudaStream_t stream) {
    if (nbEle == 0 || cmpSize == 0) return;

    //translated comment
    int maxBlocks = (nbEle + 1024) / 1025;
    if(maxBlocks<=0)
    {
        return;
    }
    //translated comment
    int* d_offsets;
    int* d_numBlocks;

    cudaError_t err1 = cudaMallocAsync(&d_offsets, (maxBlocks) * sizeof(int), stream);
    cudaError_t err2 = cudaMallocAsync(&d_numBlocks, sizeof(int), stream);

    if (err1 != cudaSuccess || err2 != cudaSuccess) {
        std::cerr << "CUDA malloc error:" << std::endl;
        std::cerr << "  d_offsets: " << cudaGetErrorString(err1) << std::endl;
        std::cerr << "  d_numBlocks: " << cudaGetErrorString(err2) << std::endl;
        
        if (err1 == cudaSuccess) cudaFreeAsync(d_offsets, stream);
        if (err2 == cudaSuccess) cudaFreeAsync(d_numBlocks, stream);
        return;
    }

    //translated comment
    cudaMemsetAsync(d_numBlocks, 0, sizeof(int), stream);

    //GPU
    calculateOffsetsKernel<<<1, 1, 0, stream>>>(
        d_cmpBytes, d_offsets, d_numBlocks, cmpSize, nbEle);

    //translated comment
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        // std::cerr << "Offset calculation kernel error: " << cudaGetErrorString(kernelErr) << std::endl;
        cudaFreeAsync(d_offsets, stream);
        cudaFreeAsync(d_numBlocks, stream);
        return;
    }
    //translated comment
    int threadsPerBlock = 128;
    int blocksPerGrid = (maxBlocks+ threadsPerBlock - 1) / threadsPerBlock;

    decompressKernelOptimized<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_cmpBytes, d_decData, d_offsets, nbEle);

    //translated comment
    cudaFreeAsync(d_offsets, stream);
    cudaFreeAsync(d_numBlocks, stream);

    //translated comment
    kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        std::cerr << "Decompression kernel error: " << cudaGetErrorString(kernelErr) << std::endl;
    }
}


//uint64_t
uint64_t bytesToULong(const unsigned char* bytes) {
    uint64_t val = 0;
    for(int i = 0; i < 8; i++) {
        val |= ((uint64_t)bytes[i]) << (i * 8);
    }
    return val;
}

//int64_t
int64_t bytesToLong(const unsigned char* bytes) {
    int64_t val = 0;
    for(int i = 0; i < 8; i++) {
        val |= ((int64_t)bytes[i]) << (i * 8);
    }
    return val;
}

//ZigZag
int64_t zigzag_decode1(uint64_t n) {
    return (n >> 1) ^ -(n & 1);
}

//translated comment
double bitsToDoubleHost(uint64_t bits) {
    double d;
    std::memcpy(&d, &bits, sizeof(d));
    return d;
}
//translated comment

//， CPU-GPU ，
void FalconDecompressor::decompress(const std::vector<unsigned char>& compressedData, std::vector<double>& output, int numDatas) {
    size_t dataSize = compressedData.size();
    if (dataSize == 0 || numDatas <= 0) {
        output.clear();
        return;
    }

    cudaEvent_t kernal_start_event,kernal_end_event;
    cudaEventCreate(&kernal_start_event);
    cudaEventCreate(&kernal_end_event);

    //offsets
    std::vector<int> offsets;
    offsets.reserve((numDatas + 1024) / 1025); //translated comment
    
    BitReader reader(compressedData);
    size_t totalBits = dataSize * 8;
    
    //translated comment
    while (reader.getBitPos() + 64 + 64 + 8 + 8 + 64 <= totalBits) {
        offsets.push_back(reader.getBitPos() / 8);
        uint64_t bitSize = reader.readBits(64);
        
        if (bitSize < 64) break;
        
        //translated comment
        if (reader.getBitPos() + bitSize - 64 > totalBits) break;
        
        reader.advance(bitSize - 64);
    }

    int numBlocks = offsets.size();
    if (numBlocks == 0) {
        output.assign(numDatas, 0.0);
        return;
    }

    //CUDA （ ）
    unsigned char* d_compressedData;
    double* d_output;
    int* d_offsets;

    //translated comment
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc(&d_compressedData, compressedData.size());
    cudaMalloc(&d_output, numDatas * sizeof(double));
    cudaMalloc(&d_offsets, offsets.size() * sizeof(int));

    //translated comment
    cudaMemcpyAsync(d_compressedData, compressedData.data(), compressedData.size(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice, stream);

    //translated comment
    int threadsPerBlock = 128; //translated comment
    int blocksPerGrid = (numBlocks + threadsPerBlock - 1) / threadsPerBlock;



    cudaEventRecord(kernal_start_event,stream);
    //translated comment
    decompressKernelOptimized<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_compressedData, d_output, d_offsets,numDatas);
    cudaEventRecord(kernal_end_event,stream);

    //translated comment
    cudaEventSynchronize(kernal_end_event);
    float totalTime;
    cudaEventElapsedTime(&totalTime, kernal_start_event, kernal_end_event);
    printf("\n解压核函数运行时间：%f\n",totalTime);

    //translated comment
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(kernelErr) << std::endl;
    }

    output.resize(numDatas);
    cudaMemcpyAsync(output.data(), d_output, numDatas * sizeof(double), cudaMemcpyDeviceToHost, stream);
    
    //translated comment
    cudaStreamSynchronize(stream);
    
    //translated comment
    cudaFree(d_compressedData);
    cudaFree(d_output);
    cudaFree(d_offsets);
    cudaStreamDestroy(stream);
    
    // std::cout << "Decompressed " << numBlocks << " blocks, " << numDatas << " elements" << std::endl;
}

//Falcon_decompress ( ， ， ）
void FalconDecompressor::Falcon_decompress(double* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, cudaStream_t stream) {
    if (nbEle == 0 || cmpSize == 0) return;

    cudaEvent_t kernal_start_event,kernal_end_event;
    cudaEventCreate(&kernal_start_event);
    cudaEventCreate(&kernal_end_event);

    //translated comment
    static thread_local std::vector<unsigned char> hostCmpBytes;
    hostCmpBytes.resize(cmpSize);

    //translated comment
    cudaError_t err = cudaMemcpyAsync(hostCmpBytes.data(), d_cmpBytes, cmpSize, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    cudaStreamSynchronize(stream);

    //translated comment
    std::vector<int> offsets;
    offsets.reserve((nbEle + 1024) / 1025);

    BitReader reader(hostCmpBytes);
    size_t totalBits = cmpSize * 8;
    size_t minHeaderSize = 64 + 64 + 8 + 8 + 64; // 192 bits

    while (reader.getBitPos() + minHeaderSize <= totalBits && offsets.size() * 1025 < nbEle) {
        offsets.push_back(reader.getBitPos() / 8);
        uint64_t bitSize = reader.readBits(64);

        if (bitSize < 64) break;

        size_t nextPos = reader.getBitPos() + bitSize - 64;
        if (nextPos > totalBits) break;

        reader.advance(bitSize - 64);
    }

    int numBlocks = offsets.size();
    if (numBlocks == 0) {
        //translated comment
        cudaMemsetAsync(d_decData, 0, nbEle * sizeof(double), stream);
        return;
    }

    //translated comment
    int* d_offsets;
    cudaMalloc(&d_offsets, offsets.size() * sizeof(int));
    cudaMemcpyAsync(d_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice, stream);

    //translated comment
    int threadsPerBlock = 128;
    int blocksPerGrid = (numBlocks + threadsPerBlock - 1) / threadsPerBlock;
    
    cudaEventRecord(kernal_start_event,stream);
    decompressKernelOptimized<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_cmpBytes, d_decData, d_offsets,  nbEle);

    cudaEventRecord(kernal_end_event,stream);

    //translated comment
    cudaEventSynchronize(kernal_end_event);
    float totalTime;
    cudaEventElapsedTime(&totalTime, kernal_start_event, kernal_end_event);
    printf("\n解压核函数运行时间：%f\n",totalTime);
    //translated comment
    cudaFree(d_offsets);

    //translated comment
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        std::cerr << "Kernel execution error: " << cudaGetErrorString(kernelErr) << std::endl;
    }
    cudaEventDestroy(kernal_start_event);
    cudaEventDestroy(kernal_end_event);
}

//GPU - +
__global__ void decompressKernelNoPack(
    const unsigned char* __restrict__ compressedData,
    double* __restrict__ output,
    const int* __restrict__ offsets,
    // int numBlocks,
    int numDatas
) {

    int blockId = blockIdx.x * blockDim.x + threadIdx.x;
    // if (blockId >= numBlocks) return;
    int numData = min(numDatas - blockId * 1025, 1025);
    
    if (numData <= 0) return;
    //translated comment
    const unsigned char* blockData = compressedData + offsets[blockId];
    size_t bitPos = 0;


    //translated comment
    uint64_t bitSize = readBitsDevice(blockData, bitPos, 64);
    int64_t firstValue = (int64_t)readBitsDevice(blockData, bitPos, 64);
    unsigned char maxDecimalPlaces = (unsigned char)readBitsDevice(blockData, bitPos, 8);
    unsigned char maxBeta = (unsigned char)readBitsDevice(blockData, bitPos, 8);
    unsigned char bitCount = (unsigned char)readBitsDevice(blockData, bitPos, 8);
    

    if (bitCount == 0 || bitCount > 64) {
        for (int i = 0; i < numData; i++) {
            output[blockId * 1025 + i] = 0.0;
        }
        return; //bitCount，
    }
    //delta

    //translated comment
    // uint64_t deltasZigzag[1024];
    size_t outputOffset = blockId * 1025;
    // if (outputOffset >= totalElements) return;
    
    // int actualElements = min(numDeltas + 1, (int)(totalElements - outputOffset));
    // actualElements=min(actualElements,1025);
    //translated comment
    double scale = (maxBeta > 15) ? 1.0 : 
        (maxDecimalPlaces < 16 ? POW10_TABLE[maxDecimalPlaces] : pow(10.0, maxDecimalPlaces));
    bool useDirectConversion = (maxBeta > 15);
    
    //translated comment
    if (useDirectConversion) {
        uint64_t bits = (uint64_t)firstValue;
        output[outputOffset] = decodeDoubleWithSignLast(bits);
    } else {
        output[outputOffset] = (double)firstValue / scale;
    }
    
    // 111111
    //translated comment
    int64_t prevValue = firstValue;
    for (int i = 1; i < numData; i++) {
        //delta delta
        uint64_t deltaZigzag = readBitsDevice(blockData, bitPos, bitCount);
        int64_t delta = zigzag_decode(deltaZigzag);
        
        //translated comment
        prevValue += delta;
        
        //translated comment
        if (useDirectConversion) {
            uint64_t bits = (uint64_t)prevValue;
            output[outputOffset + i] = decodeDoubleWithSignLast(bits);
        } else {
            output[outputOffset + i] = (double)prevValue / scale;
        }
    }
}

//translated comment
void FalconDecompressor::Falcon_decompress_no_pack(double* d_decData,
                                     unsigned char* d_cmpBytes,
                                     size_t nbEle,
                                     size_t cmpSize,
                                     cudaStream_t stream) {
    if (nbEle == 0 || cmpSize == 0) return;

    //translated comment
    int maxBlocks = (nbEle + 1024) / 1025;
    if(maxBlocks<=0)
    {
        return;
    }
    //translated comment
    int* d_offsets;
    int* d_numBlocks;

    cudaError_t err1 = cudaMallocAsync(&d_offsets, (maxBlocks) * sizeof(int), stream);
    cudaError_t err2 = cudaMallocAsync(&d_numBlocks, sizeof(int), stream);

    if (err1 != cudaSuccess || err2 != cudaSuccess) {
        std::cerr << "CUDA malloc error" << std::endl;
        if (err1 == cudaSuccess) cudaFreeAsync(d_offsets, stream);
        if (err2 == cudaSuccess) cudaFreeAsync(d_numBlocks, stream);
        return;
    }

    //translated comment
    cudaMemsetAsync(d_numBlocks, 0, sizeof(int), stream);

    //GPU
    calculateOffsetsKernel<<<1, 1, 0, stream>>>(
        d_cmpBytes, d_offsets, d_numBlocks, cmpSize, nbEle);

    //translated comment
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        // std::cerr << "Offset calculation kernel error: " << cudaGetErrorString(kernelErr) << std::endl;
        cudaFreeAsync(d_offsets, stream);
        cudaFreeAsync(d_numBlocks, stream);
        return;
    }
    //translated comment
    int threadsPerBlock = 128;
    int blocksPerGrid = (maxBlocks+ threadsPerBlock - 1) / threadsPerBlock;

    decompressKernelNoPack<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_cmpBytes, d_decData, d_offsets, nbEle);

    //translated comment
    cudaFreeAsync(d_offsets, stream);
    cudaFreeAsync(d_numBlocks, stream);

    //translated comment
    kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        std::cerr << "Decompression kernel error: " << cudaGetErrorString(kernelErr) << std::endl;
    }
}


// ------------------------------------------------------------------------------------------------------------------------------------
//translated comment

__global__ void calculateOffsetsBatchKernel(const unsigned char* __restrict__ d_cmpBytes,
                                           int* __restrict__ d_offsets,
                                           int* __restrict__ d_numBlocks,
                                           size_t cmpSize,
                                           size_t nbEle,
                                           int batchSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    //translated comment
    if (tid >= batchSize) return;
    
    size_t startBitPos = tid * 8; //translated comment
    size_t totalBits = cmpSize * 8;
    size_t minHeaderSize = 192; // 64 + 64 + 8 + 8 + 64 bits
    
    __shared__ int localOffsets[256];
    __shared__ int localCount;
    
    if (threadIdx.x == 0) {
        localCount = 0;
    }
    __syncthreads();
    
    //translated comment
    if (startBitPos + minHeaderSize <= totalBits) {
        size_t bitPos = startBitPos;
        uint64_t bitSize = readBitsDevice(d_cmpBytes, bitPos, 64);
        
        //：bitSize
        if (bitSize >= 64 && bitSize < totalBits && 
            startBitPos + bitSize <= totalBits) {
            
            int localIdx = atomicAdd(&localCount, 1);
            if (localIdx < 256) {
                localOffsets[localIdx] = startBitPos / 8;
            }
        }
    }
    
    __syncthreads();
    
    //translated comment
    if (threadIdx.x == 0 && localCount > 0) {
        int globalStart = atomicAdd(d_numBlocks, localCount);
        for (int i = 0; i < localCount && i < 256; i++) {
            d_offsets[globalStart + i] = localOffsets[i];
        }
    }
}

//translated comment
__global__ void calculateOffsetsPatternKernel(const unsigned char* __restrict__ d_cmpBytes,
                                             int* __restrict__ d_offsets,
                                             int* __restrict__ d_numBlocks,
                                             size_t cmpSize,
                                             size_t nbEle) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    __shared__ int sharedOffsets[128];
    __shared__ int sharedCount;
    
    if (threadIdx.x == 0) {
        sharedCount = 0;
    }
    __syncthreads();
    
    //translated comment
    size_t totalBytes = cmpSize;
    size_t minHeaderBytes = 24; // 192 bits = 24 bytes minimum
    
    for (size_t bytePos = tid * 8; bytePos + minHeaderBytes < totalBytes; bytePos += stride * 8) {
        //translated comment
        //GDF bitSize
        if (bytePos + 8 < totalBytes) {
            uint64_t possibleBitSize = *((uint64_t*)(d_cmpBytes + bytePos));
            
            //：bitSize
            if (possibleBitSize >= 192 && possibleBitSize < totalBytes * 8) {
                size_t expectedNextBlock = bytePos + (possibleBitSize + 7) / 8;
                
                //translated comment
                if (expectedNextBlock < totalBytes) {
                    int localIdx = atomicAdd(&sharedCount, 1);
                    if (localIdx < 128) {
                        sharedOffsets[localIdx] = bytePos;
                    }
                }
            }
        }
    }
    
    __syncthreads();
    
    //translated comment
    if (threadIdx.x == 0 && sharedCount > 0) {
        int globalStart = atomicAdd(d_numBlocks, sharedCount);
        int maxBlocks = (nbEle + 1024) / 1025;
        
        for (int i = 0; i < min(sharedCount, 128) && globalStart + i < maxBlocks; i++) {
            d_offsets[globalStart + i] = sharedOffsets[i];
        }
    }
}

//translated comment
__global__ void pipelineOffsetCalculation(const unsigned char* __restrict__ d_cmpBytes,
                                         int* __restrict__ d_offsets,
                                         int* __restrict__ d_numBlocks,
                                         size_t cmpSize,
                                         size_t nbEle,
                                         int phase) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    //translated comment
    size_t phaseSize = cmpSize / 4;
    size_t startPos = phase * phaseSize;
    size_t endPos = min(startPos + phaseSize, cmpSize);
    
    if (startPos >= endPos) return;
    
    __shared__ int sharedOffsets[64];
    __shared__ int sharedCount;
    
    if (threadIdx.x == 0) {
        sharedCount = 0;
    }
    __syncthreads();
    
    //translated comment
    for (size_t pos = startPos + tid * 8; pos + 24 < endPos; pos += blockDim.x * 8) {
        uint64_t possibleBitSize = *((uint64_t*)(d_cmpBytes + pos));
        
        if (possibleBitSize >= 192 && possibleBitSize < cmpSize * 8) {
            int localIdx = atomicAdd(&sharedCount, 1);
            if (localIdx < 64) {
                sharedOffsets[localIdx] = pos;
            }
        }
    }
    
    __syncthreads();
    
    //translated comment
    if (threadIdx.x == 0 && sharedCount > 0) {
        int globalStart = atomicAdd(d_numBlocks, sharedCount);
        for (int i = 0; i < min(sharedCount, 64); i++) {
            d_offsets[globalStart + i] = sharedOffsets[i];
        }
    }
}
