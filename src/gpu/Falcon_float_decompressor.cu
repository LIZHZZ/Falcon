
//Falcon_float_decompressor.cuh.cu
//translated comment
//

#include "Falcon_float_decompressor.cuh"
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>

//translated comment
__constant__ float POW10_TABLE[10] = {
    1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0,
    100000000.0, 1000000000.0
};

//ZigZag -
__device__ __forceinline__ int32_t zigzag_decode(uint32_t n) {
    return (int32_t)(n >> 1) ^ -((int32_t)(n & 1));
}

//translated comment
__device__ __forceinline__ uint32_t readBitsDevice(const unsigned char* buffer, size_t& bitPos, int n) {
    if (n == 0) return 0;
    if (n > 32) n = 32; //translated comment
    
    uint32_t result = 0;
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
            uint32_t lastBits = buffer[startByte + fullBytes] & ((1 << remainingBits) - 1);
            result |= lastBits << (fullBytes * 8);
        }
    } else {
        //translated comment
        for (int i = 0; i < n; i++) {
            size_t byteIdx = (bitPos + i) / 8;
            int bitIdx = (bitPos + i) % 8;
            uint32_t bit = (buffer[byteIdx] >> bitIdx) & 1ULL;
            result |= (bit << i);
        }
    }
    
    bitPos += n;
    return result;
}

__device__ float decodeFloatWithSignLast(uint32_t value) {
    uint32_t original = (value >> 1) ^ -((int32_t)(value & 1));
    union {
        uint32_t u;
        float d;
    } val;

    val.u = original;
    return val.d;
}

//translated comment
__global__ void decompressKernelOptimized(
    const unsigned char* __restrict__ compressedData,
    float* __restrict__ output,
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
    uint32_t bitSize = readBitsDevice(blockData, bitPos, 32);
    int32_t firstValue = (int32_t)readBitsDevice(blockData, bitPos, 32);
    unsigned char maxDecimalPlaces = (unsigned char)readBitsDevice(blockData, bitPos, 8);
    unsigned char maxBeta = (unsigned char)readBitsDevice(blockData, bitPos, 8);
    unsigned char bitCount = (unsigned char)readBitsDevice(blockData, bitPos, 8);

    // if(blockId==0){
    //     printf("bitSize:%d, first: %d, maxDecimalPlaces: %d, maxBeta: %d, bitcount: %d \n",bitSize, firstValue,maxDecimalPlaces,maxBeta,bitCount);
    // }


    if (bitCount == 0 || bitCount > 32) {
        //translated comment
        for (int i = 0; i < numData; i++) {
            output[blockId * 1025 + i] = 0.0;
        }
        return;
    }

    uint32_t flag1 = readBitsDevice(blockData, bitPos, 32);
    int dataByte = (numData - 1 + 7) / 8;
    int flag2Size = (dataByte + 7) / 8;
    
    //translated comment
    uint8_t result[32][128];
    uint8_t flag2[32][128];
    
    //translated comment
    #pragma unroll
    for (int i = 0; i < 32; i++) {
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

    //delta -
    uint32_t deltasZigzag[1024];
    #pragma unroll 4
    for (int j = 0; j < numData-1; j++) {
        uint32_t delta = 0;
        int byteIndex = j / 8;
        int bitIndex = 7 - (j % 8); //translated comment
        
        for (int i = 0; i < bitCount; i++) {
            uint8_t bitValue = (result[i][byteIndex] >> bitIndex) & 1;
            delta |= ((uint32_t)bitValue << (bitCount - 1 - i));
        }
        deltasZigzag[j] = delta;
    }

    //translated comment
    int32_t prevValue = firstValue;
    
    float scale = (maxBeta > 6||maxDecimalPlaces>6) ? 1.0 : 
        (maxDecimalPlaces < 8 ? POW10_TABLE[maxDecimalPlaces] : pow(10.0, maxDecimalPlaces));
    bool useDirectConversion = (maxBeta > 6||maxDecimalPlaces>6);
    
    if (useDirectConversion) {
        uint32_t bits = (uint32_t)prevValue;
        output[blockId * 1025] = decodeFloatWithSignLast(bits);
    } else {
        output[blockId * 1025] = (float)prevValue / scale;
    }
    // output[blockId * 1024] = useDirectConversion ? 
    //     *reinterpret_cast<double*>(&firstValue) : 
    //     (double)firstValue / scale;
    
    for (int i = 1; i < numData; i++) {
        int32_t delta = zigzag_decode(deltasZigzag[i-1]);
        prevValue += delta;
        
        if (useDirectConversion) {
            uint32_t bits = (uint32_t)prevValue;
            output[blockId * 1025 + i] = decodeFloatWithSignLast(bits);
        } else {
            output[blockId * 1025 + i] = (float)prevValue / scale;
        }
    }
    // if(blockId==0){
    //     printf("prevValue:%d, first: %d, maxDecimalPlaces: %d, maxBeta: %d, bitcount: %d \n",prevValue, firstValue,maxDecimalPlaces,maxBeta,bitCount);
    // }
}

//， CPU-GPU ，
void FalconDecompressor::decompress(const std::vector<unsigned char>& compressedData, std::vector<float>& output, int numDatas) {
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
    while (reader.getBitPos() + 32 + 32 + 8 + 8 + 32 <= totalBits) {
        offsets.push_back(reader.getBitPos() / 8);
        uint64_t bitSize = reader.readBits(32);
        
        if (bitSize < 32) break;
        
        //translated comment
        if (reader.getBitPos() + bitSize - 32 > totalBits) break;
        
        reader.advance(bitSize - 32);
    }

    int numBlocks = offsets.size();
    if (numBlocks == 0) {
        output.assign(numDatas, 0.0);
        return;
    }

    //CUDA （ ）
    unsigned char* d_compressedData;
    float* d_output;
    int* d_offsets;

    //translated comment
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc(&d_compressedData, compressedData.size());
    cudaMalloc(&d_output, numDatas * sizeof(float));
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
    cudaMemcpyAsync(output.data(), d_output, numDatas * sizeof(float), cudaMemcpyDeviceToHost, stream);
    
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
void FalconDecompressor::Falcon_decompress(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, cudaStream_t stream) {
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
    size_t minHeaderSize = 32 + 32 + 8 + 8 + 32; // 192 bits

    while (reader.getBitPos() + minHeaderSize <= totalBits && offsets.size() * 1024 < nbEle) {
        offsets.push_back(reader.getBitPos() / 8);
        uint64_t bitSize = reader.readBits(32);

        if (bitSize < 32) break;

        size_t nextPos = reader.getBitPos() + bitSize - 32;
        if (nextPos > totalBits) break;

        reader.advance(bitSize - 32);
    }

    int numBlocks = offsets.size();
    if (numBlocks == 0) {
        //translated comment
        cudaMemsetAsync(d_decData, 0, nbEle * sizeof(float), stream);
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

//translated comment
__global__ void calculateOffsetsKernel(const unsigned char* d_cmpBytes,
                                      int* d_offsets,
                                      int* d_numBlocks,
                                      size_t cmpSize,
                                      size_t nbEle) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return; //translated comment

    size_t bitPos = 0;
    size_t totalBits = cmpSize * 8;
    size_t minHeaderSize = 120;//192; // 32 + 32 + 8 + 8 + 8 + 32 bits
    int offsetCount = 0;
    int maxBlocks = (nbEle + 1024) / 1025;

    while (bitPos + minHeaderSize <= totalBits && offsetCount < maxBlocks) {
        d_offsets[offsetCount] = (bitPos+7) / 8;

        uint32_t bitSize = readBitsDevice(d_cmpBytes, bitPos, 32);

        if (bitSize <32) break;
        //printf(" Chunk %d: offset=%d, size= %dbytes\n",offsetCount,d_offsets[offsetCount],(bitSize+7)/8);
        size_t nextPos = bitPos + bitSize - 32;
        if (nextPos > totalBits) break;

        bitPos = nextPos;
        offsetCount++;
    }

    *d_numBlocks = offsetCount;
}
//translated comment
void FalconDecompressor::Falcon_decompress_stream_optimized(float* d_decData,
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


//uint32_t
uint32_t bytesToUInt(const unsigned char* bytes) {
    uint32_t val = 0;
    for(int i = 0; i < 8; i++) {
        val |= ((uint32_t)bytes[i]) << (i * 8);
    }
    return val;
}

//int64_t
int32_t bytesToInt(const unsigned char* bytes) {
    int32_t val = 0;
    for(int i = 0; i < 8; i++) {
        val |= ((int32_t)bytes[i]) << (i * 8);
    }
    return val;
}

//ZigZag
int32_t zigzag_decode1(uint32_t n) {
    return (n >> 1) ^ -(n & 1);
}

//translated comment
float bitsToFloatHost(uint32_t bits) {
    float d;
    std::memcpy(&d, &bits, sizeof(d));
    return d;
}
