//
#include "Falcon_compressor.cuh"
#include <iomanip> // For formatted debug output
// Constant definitions

// pow10_table and POW_NUM_G
__constant__ double pow10_table[17] = {
    1.0,                    // 10^0
    10.0,                   // 10^1
    100.0,                  // 10^2
    1000.0,                 // 10^3
    10000.0,                // 10^4
    100000.0,               // 10^5
    1000000.0,              // 10^6
    10000000.0,             // 10^7
    100000000.0,            // 10^8
    1000000000.0,           // 10^9
    10000000000.0,          // 10^10
    100000000000.0,         // 10^11
    1000000000000.0,        // 10^12
    10000000000000.0,       // 10^13
    100000000000000.0,      // 10^14
    1000000000000000.0,     // 10^15
    10000000000000000.0     // 10^16
};

//ZigZag
// __device__ static uint64_t zigzag_encode_cuda(int64_t value) {
//     return (value << 1) ^ (value >> 63);
// }
__device__ __forceinline__ static unsigned long zigzag_encode_cuda(long value) {
    return (value << 1) ^ (value >> (sizeof(long) * 8 - 1));
}


__device__ static int getDecimalPlaces(double value,int sp) {
    double trac = value + POW_NUM_G - POW_NUM_G;
    double temp = value;

    int digits = 0;
    double td = 1;
    double deltaBound = abs(value) * pow(2, -52);
    // double deltaBound = pow(2,ilogb(temp)-52);
    while (abs(temp - trac) >= deltaBound * td && digits < 16 - sp - 1)
    {
        digits++;
        td = pow10_table[digits];
        temp = value * td;
        // double deltaBound = pow(2,ilogb(temp)-52);
        trac = temp + POW_NUM_G - POW_NUM_G;
    }
    if(round(temp)/td!=value)
    {
        digits=23;
    }
    return digits;
}

__device__ static int getDecimalPlaces_br(double v,int sp) {
    v = v < 0 ? -v : v;
    
    int i = 0;
    double scale = 1.0;
    
    // Find the smallest multiplier that turns v into an exact integer
    while (i < 17) {
        double temp = v * scale;
        
        if (round(temp) == temp) {

            return i;
        }
        i++;
        scale *= 10.0;
    }
    return 17; // Reached double-precision limit
}



// Helper: print a specified range of the buffer in hexadecimal
__device__ void print_bytes(const unsigned char* buffer, size_t start, size_t length, const char* label) {
    printf("%s: ", label);
    for (size_t i = start; i < start + length; ++i) {
        // Print each byte in hexadecimal form
        printf("%02x ", buffer[i]);
    }
    printf("\n");
}

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


__device__ inline int device_min(int a, int b) {
    return (a < b) ? a : b;
}

__device__ inline int device_max(int a, int b) {
    return (a > b) ? a : b;
}

__device__ inline uint64_t device_min_uint64(uint64_t a, uint64_t b) {
    return (a < b) ? a : b;
}

__device__ inline uint64_t device_max_uint64(uint64_t a, uint64_t b) {
    return (a > b) ? a : b;
}
__device__ long encodeDoubleWithSignLast(double x) {
    union {
        double d;
        long u;
    } val;

    val.d = x;

    return (val.u << 1) ^ (val.u >> (sizeof(long) * 8 - 1));
}

__device__ inline long double2long(double data,int maxDecimalPlaces, int maxBeta)
{

    return (maxBeta > 15 ||maxDecimalPlaces>15)
    ? (encodeDoubleWithSignLast(data))
    : static_cast<long>(round(data * pow10_table[maxDecimalPlaces]));
}


__global__ void Falcon_compress_kernel(
    const double* input,
    unsigned char* output,
    volatile unsigned int* const __restrict__ cmpOffset, //translated comment
    volatile unsigned int* const __restrict__ locOffset, //translated comment
    volatile int* const __restrict__ flag,             //， warp （
    int totalSize
)
{
    //translated comment
    __shared__ unsigned int excl_sum; //translated comment
    //__shared__ unsigned int base_idx; // warp

    //translated comment
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;                   //warp （0-31）
    const int warp = idx >> 5;                     //warp

    //translated comment
    int startIdx = idx * DATA_PER_THREAD;
    int endIdx = min(startIdx + DATA_PER_THREAD, totalSize);
    int numDatas = max(0,endIdx - startIdx);
    
    uint64_t deltas[DATA_PER_THREAD]={0};

    if((idx-1)*DATA_PER_THREAD>totalSize){
        return;
    }
    int maxDecimalPlaces = 0;
    long firstValue = 0;
    volatile int bitCount = 0;
    volatile int maxBeta = 0;

    int base_block_start_idx=0;
    //int base_block_end_idx=0;
    // int quant_chunk_idx;
    //int block_idx; // ，
    // if(numDatas<1024&&numDatas>0)
    // {
    //     printf("numDatas:%d\n",numDatas);
    // }
    long currQuant=0;
    long lorenQuant=0;
    long prevQuant=0;

    unsigned int thread_ofs = 0;

    int maxSp = -99;
    //translated comment
    #pragma unroll 8
    for (int i = 0; i < DATA_PER_THREAD; ++i) {
        if (i >= numDatas) break;

        double value = input[startIdx + i];
        double log10v = log10(fabs(value));
        int sp = floor(log10v);
        maxSp = device_max(maxSp, sp);

        int alpha = getDecimalPlaces(value, sp);  //translated comment
        maxDecimalPlaces = device_max(maxDecimalPlaces, alpha);
    }
    //translated comment
    // for (int i = 0; i < numDatas; i++) {
    //     double value =input[startIdx + i];
    //     double log10v = log10(std::abs(value));
    //     int sp = floor(log10v);
    //     maxSp = device_max(maxSp, sp);
    //double alpha = getDecimalPlaces(value, sp);//
    //     // double beta =  alpha + sp + 1;
    //     // maxBeta = device_max(maxBeta,beta);
    //     // if(alpha>maxDecimalPlaces){
    //     //     idwrong = i;
    //     // }
    //     maxDecimalPlaces = device_max(maxDecimalPlaces, alpha);
    // }

    maxBeta = maxSp + maxDecimalPlaces+1;
    
    //2. FOR + zigzag（ 4 ）
    volatile uint64_t maxDelta = 0;
    firstValue = double2long(input[startIdx], maxDecimalPlaces,maxBeta); //translated comment
    prevQuant = firstValue;//translated comment
    base_block_start_idx = startIdx + 1;

    #pragma unroll 8
    for (int i = 0; i < DATA_PER_THREAD - 1; ++i) {
        if (i >= numDatas - 1) break;

        currQuant = double2long(input[base_block_start_idx + i], maxDecimalPlaces, maxBeta); //translated comment
        lorenQuant = currQuant - prevQuant;                                                  //translated comment
        deltas[i] = zigzag_encode_cuda(lorenQuant);

        maxDelta = device_max_uint64(maxDelta, deltas[i]);
        prevQuant = currQuant;
    }
    // for(int i=0;i<numDatas-1;i++){
    //currQuant = double2long(input[base_block_start_idx+i], maxDecimalPlaces,maxBeta); //
    //lorenQuant = currQuant - prevQuant; //
    //     deltas[i] = zigzag_encode_cuda(lorenQuant);
    
    //     maxDelta = device_max_uint64(maxDelta, deltas[i]);
    //     prevQuant = currQuant;

    // }

    bitCount = maxDelta > 0 ? 64 - __clzll(maxDelta) : 1;//translated comment
    bitCount = min(bitCount, (int)MAX_BITCOUNT);

        const int numByte = (numDatas - 1 + 7) / 8;
        uint8_t result_flat[8192] = {};

    //bit-plane
    #pragma unroll 8
    for (int i = 0; i < MAX_BITCOUNT; ++i) {  //MAX_BITCOUNT bitCount
        if (i >= bitCount) break;

        int j = 0;
        while ((j + 8 + 1) < numDatas) {
            int byteIndex = j / 8;  //bit
            uint8_t currentByte = 0;
            currentByte |= (((deltas[j]     >> (bitCount - 1 - i)) & 1) << 7);
            currentByte |= (((deltas[j + 1] >> (bitCount - 1 - i)) & 1) << 6);
            currentByte |= (((deltas[j + 2] >> (bitCount - 1 - i)) & 1) << 5);
            currentByte |= (((deltas[j + 3] >> (bitCount - 1 - i)) & 1) << 4);
            currentByte |= (((deltas[j + 4] >> (bitCount - 1 - i)) & 1) << 3);
            currentByte |= (((deltas[j + 5] >> (bitCount - 1 - i)) & 1) << 2);
            currentByte |= (((deltas[j + 6] >> (bitCount - 1 - i)) & 1) << 1);
            currentByte |= (((deltas[j + 7] >> (bitCount - 1 - i)) & 1) << 0);

            result_flat[i * numByte + byteIndex] = currentByte;
            j += 8;
        }
        for (; j < (numDatas - 1); ++j) {
            int byteIndex = j / 8;  //bit
            int bitIndex  = j % 8;  //bit

            uint8_t bitVal = ((deltas[j] >> (bitCount - 1 - i)) & 1);
            result_flat[i * numByte + byteIndex] |= bitVal << (7 - bitIndex);
        }
    }
/*
        for (int i = 0; i < bitCount; ++i) {//translated comment
            int j=0;
            while((j+8+1)<numDatas)
            {
                int byteIndex = j / 8;  //bit
                uint8_t currentByte = 0;
                currentByte |= (((deltas[j] >> (bitCount - 1 - i)) & 1) << 7);
                currentByte |= (((deltas[j+1] >> (bitCount - 1 - i)) & 1) << 6);
                currentByte |= (((deltas[j+2] >> (bitCount - 1 - i)) & 1) << 5);
                currentByte |= (((deltas[j+3] >> (bitCount - 1 - i)) & 1) << 4);
                currentByte |= (((deltas[j+4] >> (bitCount - 1 - i)) & 1) << 3);
                currentByte |= (((deltas[j+5] >> (bitCount - 1 - i)) & 1) << 2);
                currentByte |= (((deltas[j+6] >> (bitCount - 1 - i)) & 1) << 1);
                currentByte |= (((deltas[j+7] >> (bitCount - 1 - i)) & 1) << 0);
                
                result_flat[i * numByte + byteIndex] = currentByte;
                j+=8;
            }
            for (; j <(numDatas -1); ++j) {//numBytes

                int byteIndex = j / 8;  //bit
                int bitIndex = j % 8;   //bit

                uint8_t bitVal = ((deltas[j] >> (bitCount - 1 - i)) & 1);
                //bit


                result_flat[i * numByte + byteIndex] |= bitVal << (7 - bitIndex);


            }

        }
*/

        //4.2 ， ， bitsize
        uint64_t bitSize =  64ULL +                 // bitsize
                            64ULL +                 // firstValue
                            8ULL +                  // maxDecimalPlaces
                            8ULL +                  // maxBeta
                            8ULL +                  // bitCount
                            64ULL;                  // flag1

        uint64_t flag1 = 0;
        uint8_t flag2[(DATA_PER_THREAD-1)];

        memset(flag2,0,sizeof(flag2));
/*
        for(int i = 0;i<bitCount;i++){
            int b0 = 0;
            int b1 = 0;

            //result_flat flag2_flat
            size_t result_row_start_offset = i * numByte;
            size_t flag2_row_start_offset = i * ((numByte + 7) / 8); //flag2
        
            for (int j = 0; j < numByte; j++) {
                uint8_t current_result_byte = result_flat[result_row_start_offset + j];

                b0 += (current_result_byte == 0);
                b1 += (current_result_byte != 0);

                int flag2_byte_idx = j / 8;
                int flag2_bit_idx = j % 8;

                // uint8_t mask = (1 << flag2_bit_idx);
                //flag2 ，
                if (current_result_byte != 0) {
                    flag2[flag2_row_start_offset + flag2_byte_idx]|= (current_result_byte != 0) << flag2_bit_idx;
                } else {
                    flag2[flag2_row_start_offset + flag2_byte_idx]&= ~((current_result_byte == 0) << flag2_bit_idx);
                }
            }
            uint64_t is_sparse = (uint64_t)(((numByte + 7) / 8 + b1) < numByte);
                flag1 |= (is_sparse << i);
                flag1 &= ~((!is_sparse) << i);
                bitSize += is_sparse ? ((numByte + 7) / 8 + b1) * 8 : 8 * numByte;
        }
*/
        #pragma unroll 8
        for (int i = 0; i < MAX_BITCOUNT; ++i) {
            if (i >= bitCount) break;

            int b0 = 0;
            int b1 = 0;

            size_t result_row_start_offset = i * numByte;
            size_t flag2_row_start_offset   = i * ((numByte + 7) / 8); //flag2

            for (int j = 0; j < numByte; j++) {
                uint8_t current_result_byte = result_flat[result_row_start_offset + j];

                b0 += (current_result_byte == 0);
                b1 += (current_result_byte != 0);

                int flag2_byte_idx = j / 8;
                int flag2_bit_idx  = j % 8;

                if (current_result_byte != 0) {
                    flag2[flag2_row_start_offset + flag2_byte_idx] |=  (1u << flag2_bit_idx);
                } else {
                    flag2[flag2_row_start_offset + flag2_byte_idx] &=
                        static_cast<uint8_t>(~(1u << flag2_bit_idx));
                }
            }

            uint64_t is_sparse = (uint64_t)(((numByte + 7) / 8 + b1) < numByte);
            flag1 |= (is_sparse << i);
            flag1 &= ~((!is_sparse) << i);
            bitSize += is_sparse ? ((numByte + 7) / 8 + b1) * 8 : 8 * numByte;
        }
            
        if(numDatas<=0)
        {
            bitSize=0;
        }
    //translated comment
        thread_ofs+=bitSize;//bitSize bit

        //5.1. Warp( ) ，
        #pragma unroll 5
        for(int i = 1; i < 32; i <<= 1)
        {
            int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
            if(lane >= i) thread_ofs += tmp;                      //translated comment
        }
        __syncthreads(); //translated comment
        // printf("thread_ofs[%d]:%d",lane,thread_ofs);

        //5.2 Warp( ) locOffset flag
        if(lane == 31||numDatas<=0)//translated comment
        {
            locOffset[warp + 1] = thread_ofs; //warp
            __threadfence();                  //translated comment
            if(warp == 0)
            {
                flag[0] = 2;                   //warp
                __threadfence();
                flag[1] = 1;                   //warp
                __threadfence();
            }
            else
            {
                flag[warp + 1] = 1;            //warp
                __threadfence();
            }
            //printf("flag[%d] ready\n",warp + 1);
        }
        __syncthreads(); //， flag

        //5.3 warp， （ ）
        if(warp > 0)
        {
            if(!lane) //warp
            {
                int lookback = warp;          //warp( )
                int loc_excl_sum = 0;         //translated comment

                while(lookback > 0)//wrap（ ）
                {
                    int status;
                    do{
                        status = flag[lookback]; //warp
                    //    printf(" loop flag[%d]:%d\n",lookback,status);
                        __threadfence();         //translated comment
                    } while(status == 0);

                    if(status == 2)
                    {
                        loc_excl_sum += cmpOffset[lookback]; //warp cmpOffset
                        __threadfence();
                        break;
                    }
                    if(status == 1)
                        loc_excl_sum += locOffset[lookback]; //warp locOffset
                    lookback--;
                    __threadfence();
                   // printf(" turn flag[%d]:%d\n",lookback,status);
                }
                //printf(" loop out warp:%d\n",warp);
                excl_sum = loc_excl_sum; //translated comment

                cmpOffset[warp] = excl_sum; //warp cmpOffset
                __threadfence();           //translated comment

                //printf("flag[%d] over1\n",warp);
                if(warp == gridDim.x - 1)
                {
                    cmpOffset[warp + 1] = cmpOffset[warp] + locOffset[warp + 1]; //warp cmpOffset
                    __threadfence();

                }
                flag[warp] = 2;             //warp
                //printf("flag[%d] over2\n",warp);
                __threadfence();
            }
        }
        else {
            //warp==0： 0（ ，block ）
            if (!lane) { excl_sum = 0; }
        }
        __syncthreads(); //， cmpOffset
        if(numDatas<=0)
        {
            if(cmpOffset[warp + 1]<=0)
            {
                cmpOffset[warp + 1] = cmpOffset[warp] + locOffset[warp + 1];
            }
            return;
        }
        //translated comment
        int outputIdxBit = excl_sum + thread_ofs - bitSize; //bit wrap +wrap
        int outputIdx = (outputIdxBit+7)/8;

    //translated comment

        memcpy(output + outputIdx, &bitSize, sizeof(unsigned long long));

        //6.2. firstValue (8 )
        //firstValue (double) output
        memcpy(output + outputIdx + 8, &firstValue, sizeof(double)); //sizeof(double)
            

        // }
        //6.3. maxDecimalPlaces bitCount ( 1 )
        output[outputIdx + 16] = static_cast<unsigned char>(maxDecimalPlaces);
        output[outputIdx + 17] = static_cast<unsigned char>(maxBeta);
        output[outputIdx + 18] = static_cast<unsigned char>(bitCount);

        memcpy(output + outputIdx + 19, &flag1, sizeof(unsigned long long));
        // printf("In %d  flag1 is : %llx\n",idx,flag1);
        //translated comment
        int flag2Byte = (numByte+7)/8;
        int ofs=outputIdx + 27;
        //int res=0; //byte bit
        for(int i=0;i<bitCount;i++)
        {
            size_t flag2_row_start_offset = i * flag2Byte;
            size_t result_row_start_offset = i * numByte;
            if((flag1 & (1ULL << i)) != 0){
                memcpy(output + ofs, flag2 + flag2_row_start_offset, flag2Byte);
                ofs += flag2Byte;
                for (int j = 0; j < numByte; j++) {
                    if (result_flat[result_row_start_offset + j]) {
                        output[ofs++] = result_flat[result_row_start_offset + j];
                    }
                }
            } else { //translated comment
                memcpy(output + ofs, result_flat + result_row_start_offset, numByte);
                ofs += numByte;
            }

                
        }

}


//cmpSize BYTE
void FalconCompressor::Falcon_compress(double* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, cudaStream_t stream)
{

    int bsize = cmp_tblock_size;
    int gsize = (nbEle + bsize * DATA_PER_THREAD - 1) / (bsize * DATA_PER_THREAD);
    // size_t numthread = (nbEle + cmp_chunk - 1) / (cmp_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    unsigned int glob_sync;
    cudaMallocAsync((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMemsetAsync(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMallocAsync((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMemsetAsync(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMallocAsync((void**)&d_flag, sizeof(int)*cmpOffSize,stream);
    cudaMemsetAsync(d_flag, 0, sizeof(int)*cmpOffSize,stream);
    cudaCheckError(cudaGetLastError());
    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);

    // printf("run\n");
    Falcon_compress_kernel<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, nbEle);
    //translated comment
    cudaStreamSynchronize(stream);
    cudaCheckError(cudaGetLastError());

    
    // Obtain compression ratio and move data back to CPU.  
    
        cudaMemcpyAsync(&glob_sync, (d_cmpOffset + cmpOffSize-1), sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        cudaCheckError(cudaGetLastError());
        *cmpSize = ((size_t)glob_sync+7)/8;//+ (nbEle+cmp_tblock_size*cmp_chunk-1)/(cmp_tblock_size*cmp_chunk)*(cmp_tblock_size*cmp_chunk)/32;

    cudaFreeAsync(d_cmpOffset,stream);
    cudaFreeAsync(d_locOffset,stream);
    cudaFreeAsync(d_flag,stream);
    
}

//bits
void FalconCompressor::Falcon_compress_stream(double* d_oriData, unsigned char* d_cmpBytes, unsigned int* d2h_async_totalBits_ptr, size_t nbEle, cudaStream_t stream)
{
    // Data blocking.
    int bsize = cmp_tblock_size;
    int gsize = (nbEle + bsize * DATA_PER_THREAD - 1) / (bsize * DATA_PER_THREAD);
    // size_t numthread = (nbEle + cmp_chunk - 1) / (cmp_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    cudaMallocAsync((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMemsetAsync(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMallocAsync((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMemsetAsync(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMallocAsync((void**)&d_flag, sizeof(int)*cmpOffSize,stream);
    cudaMemsetAsync(d_flag, 0, sizeof(int)*cmpOffSize,stream);


    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);

    Falcon_compress_kernel<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, nbEle);

    // Obtain compression ratio and move data back to CPU.  
    cudaMemcpyAsync(d2h_async_totalBits_ptr, (d_cmpOffset + cmpOffSize-1), sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaFreeAsync(d_cmpOffset,stream);
    cudaFreeAsync(d_locOffset,stream);
    cudaFreeAsync(d_flag,stream);
}


__global__ void compressBlockKernel(
    const double* input,
    int totalSize,
    unsigned char* output,
    uint64_t* bitSizes,
    volatile unsigned int* const __restrict__ cmpOffset, //translated comment
    volatile unsigned int* const __restrict__ locOffset, //translated comment
    volatile int* const __restrict__ flag             //， warp （
)
{
        //translated comment
    __shared__ unsigned int excl_sum; //translated comment

    //translated comment
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;                   //warp （0-31）
    const int warp = idx >> 5;                     //warp

    //translated comment
    int startIdx = idx * DATA_PER_THREAD;
    int endIdx = min(startIdx + DATA_PER_THREAD, totalSize);
    int numDatas = endIdx - startIdx;
    uint64_t deltas[DATA_PER_THREAD];
    //int numDeltas = numDatas - 1;
    if(numDatas<=0)
    {
        return;
    }
    //translated comment
    int maxDecimalPlaces = 0;
    int maxBeta =0;
    long firstValue = 0;
    int bitCount = 0;

    int base_block_start_idx, base_block_end_idx;
    int quant_chunk_idx;
    //int block_idx; // ，

    long currQuant;
    long lorenQuant;
    long prevQuant;

    unsigned int thread_ofs = 0;
    double4 tmp_buffer;

    //translated comment
    for (int i = 0; i < numDatas; i++) {
        double value =input[startIdx + i];
        double log10v = log10(std::abs(value));
        int sp = floor(log10v);

        double alpha = getDecimalPlaces(value, sp);//translated comment
        double beta =  alpha + sp + 1;
        maxBeta = device_max(maxBeta,beta);
        maxDecimalPlaces = device_max(maxDecimalPlaces, alpha);
    }
    
    uint64_t maxDelta = 0;
    firstValue = double2long(input[startIdx], maxDecimalPlaces,maxBeta); //translated comment
    prevQuant = firstValue;                                              //translated comment
    for (int j = 0; j < (numDatas + 30) / 32; j++)
    {                                                       //translated comment
        base_block_start_idx = startIdx + j * 32 + 1;       //translated comment
        base_block_end_idx = base_block_start_idx + 32 + 1; //translated comment

        if (base_block_end_idx < totalSize)
        {
            int i = base_block_start_idx;
            #pragma unroll 8 //translated comment
            for(; i < base_block_end_idx; i += 4) {

                tmp_buffer = reinterpret_cast<const double4*>(input+1)[(i-1) / 4];
                quant_chunk_idx = j * 32 + ((i-1) % 32); //translated comment

                currQuant = double2long(tmp_buffer.x, maxDecimalPlaces,maxBeta); //translated comment
                lorenQuant = currQuant - prevQuant; //translated comment

                deltas[quant_chunk_idx] = zigzag_encode_cuda(lorenQuant);
                prevQuant = currQuant; //translated comment
                maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx]); //translated comment
                // }

                //translated comment
                currQuant = double2long(tmp_buffer.y, maxDecimalPlaces,maxBeta);
                lorenQuant = currQuant - prevQuant;

                deltas[quant_chunk_idx + 1] = zigzag_encode_cuda(lorenQuant);
                prevQuant = currQuant;
                maxDelta = device_max_uint64(maxDelta, deltas[ quant_chunk_idx + 1]);

                //translated comment
                currQuant = double2long(tmp_buffer.z, maxDecimalPlaces,maxBeta);
                lorenQuant = currQuant - prevQuant;

                deltas[quant_chunk_idx + 2] = zigzag_encode_cuda(lorenQuant);
                prevQuant = currQuant;
                maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx + 2]);

                //translated comment
                currQuant = double2long(tmp_buffer.w, maxDecimalPlaces,maxBeta);
                lorenQuant = currQuant - prevQuant;

                deltas[quant_chunk_idx + 3] = zigzag_encode_cuda(lorenQuant);
                prevQuant = currQuant;
                maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx + 3]);
            }
        }
        else {
            //translated comment
            if(base_block_start_idx >= endIdx) {
                //， absQuant 0
                quant_chunk_idx = j * 32 + (base_block_start_idx % 32);
                for(int i = quant_chunk_idx; i < quant_chunk_idx + 32; i++)
                    deltas[i] = 0;
            }
            else {
                //translated comment
                int remainbEle = totalSize - base_block_start_idx;  //translated comment
                int zeronbEle = base_block_end_idx - totalSize;     //translated comment

                //translated comment
                for (int i = base_block_start_idx; i < base_block_start_idx + remainbEle; i++)
                {
                    if (i == startIdx)
                    {
                        deltas[0]=0;
                        continue;
                    }
                    quant_chunk_idx = j * 32 + (i % 32);
                    currQuant = double2long(input[i], maxDecimalPlaces,maxBeta);

                    lorenQuant = currQuant - prevQuant;

                    deltas[ quant_chunk_idx] = zigzag_encode_cuda(lorenQuant);
                    prevQuant = currQuant;
                    maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx]);
                }

                quant_chunk_idx = j * 32 + (totalSize % 32);
                for(int i = quant_chunk_idx; i < quant_chunk_idx + zeronbEle; i++)
                    deltas[i] = 0;
            }
        }
    }

    bitCount = maxDelta > 0 ? 64 - __clzll(maxDelta) : 1;//translated comment
    bitCount = min(bitCount, (int)MAX_BITCOUNT);

        int numByte = (numDatas-1 + 7) / 8;
        // uint8_t result[64][128];
        uint8_t result[64][128] = {}; //translated comment
        //translated comment

        //uint64_t
        for (int i = 0; i < bitCount; ++i) {//translated comment
            int j=0;
            while(j+8<numDatas-1)//translated comment
            {
                int byteIndex = j / 8;  //bit
                result[i][byteIndex] = result[i][byteIndex] |
                                        (((deltas[j] >> (bitCount - 1 - i)) & 1) << (7))|
                                        (((deltas[j+1] >> (bitCount - 1 - i)) & 1) << (6))|
                                        (((deltas[j+2] >> (bitCount - 1 - i)) & 1) << (5))|
                                        (((deltas[j+3] >> (bitCount - 1 - i)) & 1) << (4))|
                                        (((deltas[j+4] >> (bitCount - 1 - i)) & 1) << (3))|
                                        (((deltas[j+5] >> (bitCount - 1 - i)) & 1) << (2))|
                                        (((deltas[j+6] >> (bitCount - 1 - i)) & 1) << (1))|
                                        (((deltas[j+7] >> (bitCount - 1 - i)) & 1) << (0));
                j+=8;
            }
            for (; j <numDatas-1 ; ++j) {//numBytes
                int byteIndex = j / 8;  //bit
                int bitIndex = j % 8;   //bit

                //bit
                result[i][byteIndex] |= (((deltas[j] >> (bitCount - 1 - i)) & 1) << (7 - bitIndex));
            }

        }

        //4.2 ， ， bitsize
        uint64_t bitSize =  64ULL +                 // bitsize
                            64ULL +                 // firstValue
                            8ULL +                  // maxDecimalPlaces
                            8ULL +                  // maxBeta
                            8ULL +                  // bitCount
                            64ULL;                  // flag1

        uint64_t flag1 = 0;              //translated comment
        uint8_t flag2[64][16];          //, 1024 ， 1024bit， 128byte,
        memset(flag2, 0, sizeof(flag2));
        int BITS_PER_THREAD=4;
        for (int i = 0; i < bitCount; i += BITS_PER_THREAD)
        { //translated comment
            for (int b = 0; b < BITS_PER_THREAD && (i + b) < bitCount; ++b)
            {
                int bit = i + b;
                int b0 = 0;
                int b1 = 0;
                for(int j = 0; j < numByte; j++) {
                    int m_byte = j / 8;
                    int m_bit = j % 8;
                    uint8_t current_result = result[bit][j];
                    b0 += (current_result == 0);
                    b1 += (current_result != 0);
                    flag2[bit][m_byte] |= (current_result != 0) << m_bit;//translated comment
                    flag2[bit][m_byte] &= ~((current_result == 0) << m_bit);//translated comment
                }
                //translated comment
                uint64_t is_sparse = ((numByte + 7) / 8 + b1) < numByte;
                flag1 |= (is_sparse << bit);
                flag1 &= ~((!is_sparse) << bit);
                bitSize += is_sparse ? ((numByte + 7) / 8 + b1) * 8 : 8 * numByte;
                //flag2 +b1 numByte*8
            }
        }
    //translated comment
        thread_ofs+=bitSize;//bitSize bit

        //5.1. Warp( ) ，
        #pragma unroll 5
        for(int i = 1; i < 32; i <<= 1)
        {
            int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
            if(lane >= i) thread_ofs += tmp;                      //translated comment
        }
        __syncthreads(); //translated comment

        //5.2 Warp( ) locOffset flag
        if(lane == 31)
        {
            locOffset[warp + 1] = thread_ofs; //warp
            __threadfence();                  //translated comment
            if(warp == 0)
            {
                flag[0] = 2;                   //warp
                __threadfence();
                flag[1] = 1;                   //warp
                __threadfence();
            }
            else
            {
                flag[warp + 1] = 1;            //warp
                __threadfence();
            }
        }
        __syncthreads(); //， flag

        //5.3 warp， （ ）
        if(warp > 0)
        {
            if(!lane) //warp
            {
                int lookback = warp;          //warp( )
                int loc_excl_sum = 0;         //translated comment

                while(lookback > 0)//wrap（ ）
                {
                    int status;
                    do{
                        status = flag[lookback]; //warp
                        __threadfence();         //translated comment
                    } while(status == 0);

                    if(status == 2)
                    {
                        loc_excl_sum += cmpOffset[lookback]; //warp cmpOffset
                        __threadfence();
                        break;
                    }
                    if(status == 1)
                        loc_excl_sum += locOffset[lookback]; //warp locOffset
                    lookback--;
                    __threadfence();
                }
                excl_sum = loc_excl_sum; //translated comment
                //2.3 cmpOffset
                cmpOffset[warp] = excl_sum; //warp cmpOffset
                __threadfence();           //translated comment

                if(warp == gridDim.x - 1)
                {
                    cmpOffset[warp + 1] = cmpOffset[warp] + locOffset[warp + 1]; //warp cmpOffset
                    __threadfence();
                }
                flag[warp] = 2;             //warp
                __threadfence();
            }
        }
        else {
            //warp==0： 0（ ，block ）
            if (!lane) { excl_sum = 0; }
        }
        __syncthreads(); //， cmpOffset

        //translated comment
        int outputIdxBit = excl_sum + thread_ofs - bitSize; //bit wrap +wrap
        int outputIdx = (outputIdxBit+7)/8;

        bitSizes[idx] = bitSize;

        unsigned long long firstValueBits = 0;
        memcpy(&firstValueBits, &firstValue, sizeof(long));
        //6.1 bitSize (8 )
            for(int i = 0; i < 8; i++) {
                output[outputIdx + i] = (bitSize >> (i * 8)) & 0xFF;

            }


            //6.2. firstValue (8 )
            for(int i = 0; i < 8; i++) {
                output[outputIdx + 8 + i] = (firstValueBits >> (i * 8)) & 0xFF;

            }
        //6.3. maxDecimalPlaces bitCount ( 1 )
        output[outputIdx + 16] = static_cast<unsigned char>(maxDecimalPlaces);
        output[outputIdx + 17] = static_cast<unsigned char>(maxBeta);
        output[outputIdx + 18] = static_cast<unsigned char>(bitCount);

        //6.4 flag1(8 )
        for(int i = 0; i < 8; i++) {
            output[outputIdx + 19 + i] = (flag1 >> (i * 8)) & 0xFF;
        }
        //translated comment
        int flag2Byte=(numByte+7)/8;
        int ofs=outputIdx + 27;
        for(int i=0;i<bitCount;i++)
        {
            if((flag1 & (1ULL << i)) != 0)//flag i bit 0:
            {
                //6.5.1 flag2+data
                for(int j=0;j<flag2Byte;j++)
                {
                    output[ofs++] = static_cast<unsigned char>(flag2[i][j]);
                }
                for(int j=0;j<numByte;j++)
                {
                    if(result[i][j])
                    {
                        output[ofs++] = static_cast<unsigned char>(result[i][j]);
                    }
                }
            }
            else{
                //6.5.2 data

                for(int j=0;j<numByte;j++)
                {
                    output[ofs++] = static_cast<unsigned char>(result[i][j]);
                }
            }

        }
}


//translated comment
void FalconCompressor::setupDeviceMemory(
    const std::vector<double>& input,
    double*& d_input,
    unsigned char*& d_output,
    uint64_t*& d_bitSizes
) {
    size_t inputSize = input.size();
    int numBlocks = (inputSize + DATA_PER_THREAD - 1) / (DATA_PER_THREAD);

    //translated comment
    cudaCheckError(cudaMalloc((void**)&d_input, inputSize * sizeof(double)));
    cudaCheckError(cudaMemcpy(d_input, input.data(), inputSize * sizeof(double), cudaMemcpyHostToDevice));

    //translated comment
    cudaCheckError(cudaMalloc((void**)&d_output, numBlocks * MAX_BYTES_PER_BLOCK * sizeof(unsigned char)));

    //bitSizes
    cudaCheckError(cudaMalloc((void**)&d_bitSizes, numBlocks * sizeof(uint64_t)));

}

//translated comment
void FalconCompressor::freeDeviceMemory(
    double* d_input,
    unsigned char* d_output,
    uint64_t* d_bitSizes
) {
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_bitSizes);
}


//translated comment
void FalconCompressor::compress(const std::vector<double>& input, std::vector<unsigned char>& output) {
    //std::cout<<"begin1\n";
    size_t inputSize = input.size();
    if (inputSize == 0) return;


    int blockSize = BLOCK_SIZE_G; //translated comment
    size_t numBlocks = (inputSize + blockSize * DATA_PER_THREAD - 1) / (blockSize * DATA_PER_THREAD); //translated comment
    size_t numthread = (inputSize + DATA_PER_THREAD - 1) / (DATA_PER_THREAD); //translated comment
    double* d_input = nullptr;
    unsigned char* d_output = nullptr;
    uint64_t* d_bitSizes = nullptr;

    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    int cmpOffSize = numBlocks + 1;
    cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);

    cudaMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);

    cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);
    //translated comment
    setupDeviceMemory(input, d_input, d_output, d_bitSizes);


    size_t sharedMemSize = 64; //SharedMemory
    //std::cout<<"begin2\n";

    //CUDA
    cudaEvent_t start, stop;
    cudaCheckError(cudaEventCreate(&start));
    cudaCheckError(cudaEventCreate(&stop));

    //translated comment
    cudaCheckError(cudaEventRecord(start));
    
    //translated comment
    compressBlockKernel<<<numBlocks, blockSize, sharedMemSize>>>(
        d_input,
        inputSize,
        d_output,
        d_bitSizes,
        d_cmpOffset,
        d_locOffset,
        d_flag
    );
    //translated comment
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());
    //std::cout<<"end2\n";

    //translated comment
    cudaCheckError(cudaEventRecord(stop));
    cudaCheckError(cudaEventSynchronize(stop)); //translated comment

    //translated comment
    float milliseconds = 0;
    cudaCheckError(cudaEventElapsedTime(&milliseconds, start, stop));

    //translated comment
    size_t dataSizeBytes = input.size() * sizeof(double); //translated comment
    float seconds = milliseconds / 1000.0f;

    // MB/s = (bytes / 1e6) / seconds
    float throughputMBs = (dataSizeBytes / 1e6) / seconds; 
    
    // GB/s = (bytes / 1e9) / seconds
    float throughputGBs = (dataSizeBytes / 1e9) / seconds;

    //translated comment
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "压缩核函数时间" << milliseconds
              << " ms. \nThroughput: "
              << throughputMBs << " MB/s ("
              << throughputGBs << " GB/s)" 
              << std::endl;

    //translated comment
    cudaCheckError(cudaEventDestroy(start));
    cudaCheckError(cudaEventDestroy(stop));


    //bitSizes

    std::vector<uint64_t> bitSizes(numthread);
    cudaCheckError(cudaMemcpy(bitSizes.data(), d_bitSizes, numthread * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    std::vector<uint64_t> offsets(numthread, 0);
    uint64_t totalCompressedBits = 0;
    for (size_t i = 0; i < numthread; i++) {
        offsets[i] = totalCompressedBits;
        totalCompressedBits += bitSizes[i];

    }

    uint64_t totalCompressedBytes = (totalCompressedBits + 7) / 8; //translated comment

    //translated comment
    output.resize(totalCompressedBytes, 0);
    //d_output
    std::vector<unsigned char> tempOutput(totalCompressedBytes);
    cudaCheckError(cudaMemcpy(tempOutput.data(), d_output,totalCompressedBytes * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    output = std::move(tempOutput);

    freeDeviceMemory(d_input, d_output, d_bitSizes);
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}


__global__ void Falcon_compress_kernel_no_pack(
    const double* input,
    unsigned char* output,
    volatile unsigned int* const __restrict__ cmpOffset, //translated comment
    volatile unsigned int* const __restrict__ locOffset, //translated comment
    volatile int* const __restrict__ flag,             //， warp （
    int totalSize
)
{
    //translated comment
    __shared__ unsigned int excl_sum; //translated comment

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;                   //warp （0-31）
    const int warp = idx >> 5;                     //warp

    //translated comment
    int startIdx = idx * DATA_PER_THREAD;
    int endIdx = min(startIdx + DATA_PER_THREAD, totalSize);
    int numDatas = max(0,endIdx - startIdx);
    
    uint64_t deltas[DATA_PER_THREAD]={0};

    if((idx-1)*DATA_PER_THREAD>totalSize){
        return;
    }
    int maxDecimalPlaces = 0;
    long firstValue = 0;
    volatile int bitCount = 0;
    volatile int maxBeta = 0;

    int base_block_start_idx=0;

    long currQuant=0;
    long lorenQuant=0;
    long prevQuant=0;

    unsigned int thread_ofs = 0;

    int maxSp = -99;
    //translated comment
    for (int i = 0; i < numDatas; i++) {
        double value =input[startIdx + i];
        double log10v = log10(std::abs(value));
        int sp = floor(log10v);
        maxSp = device_max(maxSp, sp);
        double alpha = getDecimalPlaces(value, sp);//translated comment
        // double beta =  alpha + sp + 1;
        // maxBeta = device_max(maxBeta,beta);
        // if(alpha>maxDecimalPlaces){
        //     idwrong = i;
        // }
        maxDecimalPlaces = device_max(maxDecimalPlaces, alpha);
    }

    maxBeta = maxSp + maxDecimalPlaces+1;

    //2. FOR + zigzag（ 4 ）
    volatile uint64_t maxDelta = 0;
    firstValue = double2long(input[startIdx], maxDecimalPlaces,maxBeta); //translated comment
    prevQuant = firstValue;//translated comment
    base_block_start_idx = startIdx + 1;

    for(int i=0;i<numDatas-1;i++){
        currQuant = double2long(input[base_block_start_idx+i], maxDecimalPlaces,maxBeta); //translated comment
        lorenQuant = currQuant - prevQuant; //translated comment
        deltas[i] = zigzag_encode_cuda(lorenQuant);
    
        maxDelta = device_max_uint64(maxDelta, deltas[i]);
        prevQuant = currQuant;

    }

    bitCount = maxDelta > 0 ? 64 - __clzll(maxDelta) : 1;//translated comment
    bitCount = min(bitCount, (int)MAX_BITCOUNT);

        const int numByte = (numDatas - 1 + 7) / 8;
        uint8_t result_flat[8192] = {};

        for (int i = 0; i < bitCount; ++i) {//translated comment
            int j=0;
            while((j+8+1)<numDatas)
            {
                int byteIndex = j / 8;  //bit
                uint8_t currentByte = 0;
                currentByte |= (((deltas[j] >> (bitCount - 1 - i)) & 1) << 7);
                currentByte |= (((deltas[j+1] >> (bitCount - 1 - i)) & 1) << 6);
                currentByte |= (((deltas[j+2] >> (bitCount - 1 - i)) & 1) << 5);
                currentByte |= (((deltas[j+3] >> (bitCount - 1 - i)) & 1) << 4);
                currentByte |= (((deltas[j+4] >> (bitCount - 1 - i)) & 1) << 3);
                currentByte |= (((deltas[j+5] >> (bitCount - 1 - i)) & 1) << 2);
                currentByte |= (((deltas[j+6] >> (bitCount - 1 - i)) & 1) << 1);
                currentByte |= (((deltas[j+7] >> (bitCount - 1 - i)) & 1) << 0);
                
                result_flat[i * numByte + byteIndex] = currentByte;
                j+=8;
            }
            for (; j <(numDatas -1); ++j) {//numBytes

                int byteIndex = j / 8;  //bit
                int bitIndex = j % 8;   //bit

                uint8_t bitVal = ((deltas[j] >> (bitCount - 1 - i)) & 1);
                //bit


                result_flat[i * numByte + byteIndex] |= bitVal << (7 - bitIndex);


            }

        }


        //4.2 ， ， bitsize
        uint64_t bitSize =  64ULL +                 // bitsize
                            64ULL +                 // firstValue
                            8ULL +                  // maxDecimalPlaces
                            8ULL +                  // maxBeta
                            8ULL +                  // bitCount
                            64ULL;                  // flag1

        uint64_t flag1 = 0;
        uint8_t flag2[(DATA_PER_THREAD-1)];

        memset(flag2,0,sizeof(flag2));

        for(int i = 0;i<bitCount;i++){
            int b0 = 0;
            int b1 = 0;

            //result_flat flag2_flat
            size_t result_row_start_offset = i * numByte;
            size_t flag2_row_start_offset = i * ((numByte + 7) / 8); //flag2
        
            for (int j = 0; j < numByte; j++) {
                uint8_t current_result_byte = result_flat[result_row_start_offset + j];

                b0 += (current_result_byte == 0);
                b1 += (current_result_byte != 0);

                int flag2_byte_idx = j / 8;
                int flag2_bit_idx = j % 8;

                // uint8_t mask = (1 << flag2_bit_idx);
                //flag2 ，
                if (current_result_byte != 0) {
                    flag2[flag2_row_start_offset + flag2_byte_idx]|= (current_result_byte != 0) << flag2_bit_idx;
                } else {
                    flag2[flag2_row_start_offset + flag2_byte_idx]&= ~((current_result_byte == 0) << flag2_bit_idx);
                }
            }
            uint64_t is_sparse = 0;//(uint64_t)(((numByte + 7) / 8 + b1) < numByte);
                // flag1|=(1<< i);
                flag1 |= (is_sparse << i);
                flag1 &= ~((!is_sparse) << i);
                bitSize += is_sparse ? ((numByte + 7) / 8 + b1) * 8 : 8 * numByte;
        }


        if(numDatas<=0)
        {
            bitSize=0;
        }
    //translated comment
        thread_ofs+=bitSize;//bitSize bit

        //5.1. Warp( ) ，
        #pragma unroll 5
        for(int i = 1; i < 32; i <<= 1)
        {
            int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
            if(lane >= i) thread_ofs += tmp;                      //translated comment
        }
        __syncthreads(); //translated comment
        // printf("thread_ofs[%d]:%d",lane,thread_ofs);

        //5.2 Warp( ) locOffset flag
        if(lane == 31||numDatas<=0)//translated comment
        {
            locOffset[warp + 1] = thread_ofs; //warp
            __threadfence();                  //translated comment
            if(warp == 0)
            {
                flag[0] = 2;                   //warp
                __threadfence();
                flag[1] = 1;                   //warp
                __threadfence();
            }
            else
            {
                flag[warp + 1] = 1;            //warp
                __threadfence();
            }
            //printf("flag[%d] ready\n",warp + 1);
        }
        __syncthreads(); //， flag

        //5.3 warp， （ ）
        if(warp > 0)
        {
            if(!lane) //warp
            {
                int lookback = warp;          //warp( )
                int loc_excl_sum = 0;         //translated comment

                while(lookback > 0)//wrap（ ）
                {
                    int status;
                    do{
                        status = flag[lookback]; //warp
                    //    printf(" loop flag[%d]:%d\n",lookback,status);
                        __threadfence();         //translated comment
                    } while(status == 0);

                    if(status == 2)
                    {
                        loc_excl_sum += cmpOffset[lookback]; //warp cmpOffset
                        __threadfence();
                        break;
                    }
                    if(status == 1)
                        loc_excl_sum += locOffset[lookback]; //warp locOffset
                    lookback--;
                    __threadfence();
                   // printf(" turn flag[%d]:%d\n",lookback,status);
                }
                //printf(" loop out warp:%d\n",warp);
                excl_sum = loc_excl_sum; //translated comment

                cmpOffset[warp] = excl_sum; //warp cmpOffset
                __threadfence();           //translated comment

                //printf("flag[%d] over1\n",warp);
                if(warp == gridDim.x - 1)
                {
                    cmpOffset[warp + 1] = cmpOffset[warp] + locOffset[warp + 1]; //warp cmpOffset
                    __threadfence();

                }
                flag[warp] = 2;             //warp
                //printf("flag[%d] over2\n",warp);
                __threadfence();
            }
        }
        else {
            //warp==0： 0（ ，block ）
            if (!lane) { excl_sum = 0; }
        }
        __syncthreads(); //， cmpOffset
        if(numDatas<=0)
        {
            if(cmpOffset[warp + 1]<=0)
            {
                cmpOffset[warp + 1] = cmpOffset[warp] + locOffset[warp + 1];
            }
            return;
        }
        //translated comment
        int outputIdxBit = excl_sum + thread_ofs - bitSize; //bit wrap +wrap
        int outputIdx = (outputIdxBit+7)/8;

    //translated comment

        memcpy(output + outputIdx, &bitSize, sizeof(unsigned long long));

        //6.2. firstValue (8 )
        //firstValue (double) output
        memcpy(output + outputIdx + 8, &firstValue, sizeof(double)); //sizeof(double)
            

        // }
        //6.3. maxDecimalPlaces bitCount ( 1 )
        output[outputIdx + 16] = static_cast<unsigned char>(maxDecimalPlaces);
        output[outputIdx + 17] = static_cast<unsigned char>(maxBeta);
        output[outputIdx + 18] = static_cast<unsigned char>(bitCount);

        //6.4 flag1(8 )
        memcpy(output + outputIdx + 19, &flag1, sizeof(unsigned long long));

        //translated comment
        int flag2Byte = (numByte+7)/8;
        int ofs=outputIdx + 27;
        for(int i=0;i<bitCount;i++)
        {
            size_t flag2_row_start_offset = i * flag2Byte;
            size_t result_row_start_offset = i * numByte;
            if((flag1 & (1ULL << i)) != 0){
                memcpy(output + ofs, flag2 + flag2_row_start_offset, flag2Byte);
                ofs += flag2Byte;
                for (int j = 0; j < numByte; j++) {
                    if (result_flat[result_row_start_offset + j]) {
                        output[ofs++] = result_flat[result_row_start_offset + j];
                    }
                }
            } else { //translated comment
                memcpy(output + ofs, result_flat + result_row_start_offset, numByte);
                ofs += numByte;
            }

                
        }

}


//translated comment
void FalconCompressor::Falcon_compress_no_pack(double* d_oriData, unsigned char* d_cmpBytes, unsigned int* d2h_async_totalBits_ptr, size_t nbEle, cudaStream_t stream)
{
    // Data blocking.
    int bsize = cmp_tblock_size;
    int gsize = (nbEle + bsize * DATA_PER_THREAD - 1) / (bsize * DATA_PER_THREAD);
    // size_t numthread = (nbEle + cmp_chunk - 1) / (cmp_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    cudaMallocAsync((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMemsetAsync(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMallocAsync((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMemsetAsync(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMallocAsync((void**)&d_flag, sizeof(int)*cmpOffSize,stream);
    cudaMemsetAsync(d_flag, 0, sizeof(int)*cmpOffSize,stream);


    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);

    Falcon_compress_kernel_no_pack<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, nbEle);

    // printf("beta %.d ",d_cmpBytes[16]);
    // Obtain compression ratio and move data back to CPU.  
    cudaMemcpyAsync(d2h_async_totalBits_ptr, (d_cmpOffset + cmpOffSize-1), sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaFreeAsync(d_cmpOffset,stream);
    cudaFreeAsync(d_locOffset,stream);
    cudaFreeAsync(d_flag,stream);
}

__global__ void Falcon_compress_kernel_br(
    const double* input,
    unsigned char* output,
    volatile unsigned int* const __restrict__ cmpOffset, //translated comment
    volatile unsigned int* const __restrict__ locOffset, //translated comment
    volatile int* const __restrict__ flag,             //， warp （
    int totalSize
)
{
    //translated comment
    __shared__ unsigned int excl_sum; //translated comment
    //__shared__ unsigned int base_idx; // warp

    //translated comment
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;                   //warp （0-31）
    const int warp = idx >> 5;                     //warp

    //translated comment
    int startIdx = idx * DATA_PER_THREAD;
    int endIdx = min(startIdx + DATA_PER_THREAD, totalSize);
    int numDatas = endIdx - startIdx;
    uint64_t deltas[DATA_PER_THREAD]={};

    int maxDecimalPlaces = 0;
    int maxBeta =0;
    long firstValue = 0;
    int bitCount = 0;

    int base_block_start_idx;
    // int base_block_end_idx;
    // int quant_chunk_idx;
    //int block_idx; // ，

    long currQuant=0;
    long lorenQuant=0;
    long prevQuant=0;

    unsigned int thread_ofs = 0;
    //translated comment
    for (int i = 0; i < numDatas; i++) {
        double value =input[startIdx + i];
        double log10v = log10(std::abs(value));
        int sp = floor(log10v);
        
        double alpha = getDecimalPlaces_br(value, sp);//translated comment
        double beta =  alpha + sp + 1;
        maxBeta = device_max(maxBeta,beta);
        maxDecimalPlaces = device_max(maxDecimalPlaces, alpha);
    }
    //printf("maxDecimalPlaces:%d\n", maxDecimalPlaces);
    //2. FOR + zigzag（ 4 ）
    uint64_t maxDelta = 0;
    firstValue = double2long(input[startIdx], maxDecimalPlaces,maxBeta); //translated comment
    prevQuant = firstValue;//translated comment
    base_block_start_idx = startIdx + 1;

    for(int i=0;i<numDatas-1;i++){
        currQuant = double2long(input[base_block_start_idx+i], maxDecimalPlaces,maxBeta); //translated comment
        lorenQuant = currQuant - prevQuant; //translated comment
        deltas[i] = zigzag_encode_cuda(lorenQuant);
    
        maxDelta = device_max_uint64(maxDelta, deltas[i]);
        prevQuant = currQuant;
        // if(startIdx<4){
        //     printf("input %d, ",deltas[i]);
        // }
    }
    //for(int j = 0; j < (numDatas+31) / 32; j++) { // / （32）
    //base_block_start_idx = startIdx + j * 32; // 32
    //base_block_end_idx = base_block_start_idx + 32; // 32

    //     if(base_block_end_idx < totalSize) {
    //             int i = base_block_start_idx;
                
            
    //#pragma unroll 8 // 8 ， 4*8=32 , 7 ，
    //         for(; i < base_block_end_idx; i += 4) {

    //             tmp_buffer = reinterpret_cast<const double4*>(input)[(i) / 4];
    //quant_chunk_idx = j * 32 + ((i) % 32); //

    //currQuant = double2long(tmp_buffer.x, maxDecimalPlaces,maxBeta); //
    //lorenQuant = currQuant - prevQuant; //

    //             deltas[quant_chunk_idx] = zigzag_encode_cuda(lorenQuant);
    //             //printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx],quant_chunk_idx,lorenQuant,currQuant,prevQuant);
    //prevQuant = currQuant; //
    //maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx]); //
    //             // }

    //translated comment
    //             currQuant = double2long(tmp_buffer.y, maxDecimalPlaces,maxBeta);
    //             lorenQuant = currQuant - prevQuant;

    //             deltas[quant_chunk_idx + 1] = zigzag_encode_cuda(lorenQuant);
    //             //printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx+1],quant_chunk_idx+1,lorenQuant,currQuant,prevQuant);
    //             prevQuant = currQuant;
    //             maxDelta = device_max_uint64(maxDelta, deltas[ quant_chunk_idx + 1]);

    //translated comment
    //             currQuant = double2long(tmp_buffer.z, maxDecimalPlaces,maxBeta);
    //             lorenQuant = currQuant - prevQuant;

    //             deltas[quant_chunk_idx + 2] = zigzag_encode_cuda(lorenQuant);
    //             // printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx+2],quant_chunk_idx+2,lorenQuant,currQuant,prevQuant);
    //             prevQuant = currQuant;
    //             maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx + 2]);

    //translated comment
    //             currQuant = double2long(tmp_buffer.w, maxDecimalPlaces,maxBeta);
    //             lorenQuant = currQuant - prevQuant;

    //             deltas[quant_chunk_idx + 3] = zigzag_encode_cuda(lorenQuant);
    //             // printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx+3],quant_chunk_idx+3,lorenQuant,currQuant,prevQuant);
    //             prevQuant = currQuant;
    //             maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx + 3]);
    //         }
    //     }
    //     else {
    //translated comment
    //         if(base_block_start_idx >= endIdx) {
    //// ， absQuant 0
    //             quant_chunk_idx = j * 32 + (base_block_start_idx % 32);
    //             for(int i = quant_chunk_idx; i < quant_chunk_idx + 32; i++)
    //                 deltas[i] = 0;
    //         }
    //         else {
    //translated comment
    //int remainbEle = totalSize - base_block_start_idx; //
    //int zeronbEle = base_block_end_idx - totalSize; //

    //translated comment
    //             for(int i = base_block_start_idx; i < base_block_start_idx + remainbEle; i++) {
    //                 if(i==startIdx)
    //                 {
    //                     deltas[0]=0;
    //                     continue;
    //                 }
    //                 quant_chunk_idx = j * 32 + (i % 32);
    //                 currQuant = double2long(input[i], maxDecimalPlaces,maxBeta);

    //                 lorenQuant = currQuant - prevQuant;

    //                 deltas[ quant_chunk_idx] = zigzag_encode_cuda(lorenQuant);

    //                 prevQuant = currQuant;
    //                 maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx]);
    //             }

    //             quant_chunk_idx = j * 32 + (totalSize % 32);
    //             for(int i = quant_chunk_idx; i < quant_chunk_idx + zeronbEle; i++)
    //                 deltas[i] = 0;
    //         }
    //     }
    // }

    bitCount = maxDelta > 0 ? 64 - __clzll(maxDelta) : 1;//translated comment
    bitCount = min(bitCount, (int)MAX_BITCOUNT);


        int numByte = (numDatas-1 + 7) / 8;
        // uint8_t result[64][128];
        uint8_t result[64][128] = {}; 
        //translated comment

        //uint64_t
        for (int i = 0; i < bitCount; ++i) {//translated comment
            int j=0;

            while(j+8<numDatas-1)//translated comment
            {
                int byteIndex = j / 8;  //bit
                result[i][byteIndex] = result[i][byteIndex] |
                                        (((deltas[j] >> (bitCount - 1 - i)) & 1) << (7))|
                                        (((deltas[j+1] >> (bitCount - 1 - i)) & 1) << (6))|
                                        (((deltas[j+2] >> (bitCount - 1 - i)) & 1) << (5))|
                                        (((deltas[j+3] >> (bitCount - 1 - i)) & 1) << (4))|
                                        (((deltas[j+4] >> (bitCount - 1 - i)) & 1) << (3))|
                                        (((deltas[j+5] >> (bitCount - 1 - i)) & 1) << (2))|
                                        (((deltas[j+6] >> (bitCount - 1 - i)) & 1) << (1))|
                                        (((deltas[j+7] >> (bitCount - 1 - i)) & 1) << (0));
                j+=8;
            }
            for (; j <numDatas -1; ++j) {//numBytes
                //（ bit ）
                // if(i==0)
                // {
                //printf("0x %02x ", deltas[j]); // ， 2
                // }
                int byteIndex = j / 8;  //bit
                int bitIndex = j % 8;   //bit

                //bit
                result[i][byteIndex] |= (((deltas[j] >> (bitCount - 1 - i)) & 1) << (7 - bitIndex));
            }

        }

        //4.2 ， ， bitsize
        uint64_t bitSize =  64ULL +                 // bitsize
                            64ULL +                 // firstValue
                            8ULL +                  // maxDecimalPlaces
                            8ULL +                  // maxBeta
                            8ULL +                  // bitCount
                            64ULL;                  // flag1

        uint64_t flag1 = 0;              //translated comment
        uint8_t flag2[64][16];          //, 1024 ， 1024bit， 128byte,
        memset(flag2, 0, sizeof(flag2));

        int BITS_PER_THREAD=4;
        for(int i = 0; i < bitCount; i += BITS_PER_THREAD) { //translated comment
            for(int b = 0; b < BITS_PER_THREAD && (i + b) < bitCount; ++b) {
                int bit = i + b;
                int b0 = 0;
                int b1 = 0;
                for(int j = 0; j < numByte; j++) {
                    int m_byte = j / 8;
                    int m_bit = j % 8;
                    uint8_t current_result = result[bit][j];
                    b0 += (current_result == 0);
                    b1 += (current_result != 0);
                    flag2[bit][m_byte] |= (current_result != 0) << m_bit;//translated comment
                    flag2[bit][m_byte] &= ~((current_result == 0) << m_bit);//translated comment
                }
                //translated comment
                uint64_t is_sparse = ((numByte + 7) / 8 + b1) < numByte;
                flag1 |= (is_sparse << bit);
                flag1 &= ~((!is_sparse) << bit);
                bitSize += is_sparse ? ((numByte + 7) / 8 + b1) * 8 : 8 * numByte;
                //flag2 +b1 numByte*8
            }
        }


        if(numDatas<=0)
        {
            bitSize=0;
        }
    //translated comment
        thread_ofs+=bitSize;//bitSize bit

        //5.1. Warp( ) ，
        #pragma unroll 5
        for(int i = 1; i < 32; i <<= 1)
        {
            int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
            if(lane >= i) thread_ofs += tmp;                      //translated comment
        }
        __syncthreads(); //translated comment
        // printf("thread_ofs[%d]:%d",lane,thread_ofs);

        //5.2 Warp( ) locOffset flag
        if(lane == 31||numDatas<=0)//translated comment
        {
            locOffset[warp + 1] = thread_ofs; //warp
            __threadfence();                  //translated comment
            if(warp == 0)
            {
                flag[0] = 2;                   //warp
                __threadfence();
                flag[1] = 1;                   //warp
                __threadfence();
            }
            else
            {
                flag[warp + 1] = 1;            //warp
                __threadfence();
            }
            //printf("flag[%d] ready\n",warp + 1);
        }
        __syncthreads(); //， flag

        //5.3 warp， （ ）
        if(warp > 0)
        {
            if(!lane) //warp
            {
                int lookback = warp;          //warp( )
                int loc_excl_sum = 0;         //translated comment

                while(lookback > 0)//wrap（ ）
                {
                    int status;
                    do{
                        status = flag[lookback]; //warp
                    //    printf(" loop flag[%d]:%d\n",lookback,status);
                        __threadfence();         //translated comment
                    } while(status == 0);

                    if(status == 2)
                    {
                        loc_excl_sum += cmpOffset[lookback]; //warp cmpOffset
                        __threadfence();
                        break;
                    }
                    if(status == 1)
                        loc_excl_sum += locOffset[lookback]; //warp locOffset
                    lookback--;
                    __threadfence();
                   // printf(" turn flag[%d]:%d\n",lookback,status);
                }
                //printf(" loop out warp:%d\n",warp);
                excl_sum = loc_excl_sum; //translated comment

                cmpOffset[warp] = excl_sum; //warp cmpOffset
                __threadfence();           //translated comment

                //printf("flag[%d] over1\n",warp);
                if(warp == gridDim.x - 1)
                {
                    cmpOffset[warp + 1] = cmpOffset[warp] + locOffset[warp + 1]; //warp cmpOffset
                    __threadfence();

                }
                flag[warp] = 2;             //warp
                //printf("flag[%d] over2\n",warp);
                __threadfence();
            }
        }
        else {
            //warp==0： 0（ ，block ）
            if (!lane) { excl_sum = 0; }
        }
        __syncthreads(); //， cmpOffset
        if(numDatas<=0)
        {
            if(cmpOffset[warp + 1]<=0)
            {
                cmpOffset[warp + 1] = cmpOffset[warp] + locOffset[warp + 1];
            }
            return;
        }
        //translated comment
        int outputIdxBit = excl_sum + thread_ofs - bitSize; //bit wrap +wrap
        int outputIdx = (outputIdxBit+7)/8;

    //translated comment


        unsigned long long firstValueBits = 0;
        memcpy(&firstValueBits, &firstValue, sizeof(long));
        // if (outputIdx % 8 != 0) {
        //6.1 bitSize (8 )
            for(int i = 0; i < 8; i++) {
                output[outputIdx + i] = (bitSize >> (i * 8)) & 0xFF;

            }


            //6.2. firstValue (8 )
            for(int i = 0; i < 8; i++) {
                output[outputIdx + 8 + i] = (firstValueBits >> (i * 8)) & 0xFF;

            }

        // }
        //6.3. maxDecimalPlaces bitCount ( 1 )
        output[outputIdx + 16] = static_cast<unsigned char>(maxDecimalPlaces);
        output[outputIdx + 17] = static_cast<unsigned char>(maxBeta);
        output[outputIdx + 18] = static_cast<unsigned char>(bitCount);

        //6.4 flag1(8 )
        for(int i = 0; i < 8; i++) {
            output[outputIdx + 19 + i] = (flag1 >> (i * 8)) & 0xFF;
        }
        // printf("In %d  flag1 is : %llx\n",idx,flag1);
        //translated comment
        int flag2Byte=(numByte+7)/8;
        int ofs=outputIdx + 27;
        //int res=0; //byte bit
        for(int i=0;i<bitCount;i++)
        {
            if((flag1 & (1ULL << i)) != 0)//flag i bit 0:
            {
                //6.5.1 flag2+data
                for(int j=0;j<flag2Byte;j++)
                {
                    output[ofs++] = static_cast<unsigned char>(flag2[i][j]);
                    // printf("flag2[%d][%d]:0x%llx\n",i,j,flag2[i][j]);
                }
                for(int j=0;j<numByte;j++)
                {
                    if(result[i][j])
                    {
                        output[ofs++] = static_cast<unsigned char>(result[i][j]);
                    }
                }
            }
            else{
                //6.5.2 data

                for(int j=0;j<numByte;j++)
                {
                    output[ofs++] = static_cast<unsigned char>(result[i][j]);
                }
            }

        }


}

//bits
void FalconCompressor::Falcon_compress_br(double* d_oriData, unsigned char* d_cmpBytes, unsigned int* d2h_async_totalBits_ptr, size_t nbEle, cudaStream_t stream)
{
    // Data blocking.
    int bsize = cmp_tblock_size;
    int gsize = (nbEle + bsize * DATA_PER_THREAD - 1) / (bsize * DATA_PER_THREAD);
    // size_t numthread = (nbEle + cmp_chunk - 1) / (cmp_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    cudaMallocAsync((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMemsetAsync(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMallocAsync((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMemsetAsync(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMallocAsync((void**)&d_flag, sizeof(int)*cmpOffSize,stream);
    cudaMemsetAsync(d_flag, 0, sizeof(int)*cmpOffSize,stream);


    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);

    Falcon_compress_kernel_br<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, nbEle);

    // Obtain compression ratio and move data back to CPU.  
    cudaMemcpyAsync(d2h_async_totalBits_ptr, (d_cmpOffset + cmpOffSize-1), sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaFreeAsync(d_cmpOffset,stream);
    cudaFreeAsync(d_locOffset,stream);
    cudaFreeAsync(d_flag,stream);
}


__global__ void Falcon_compress_kernel_spare(
    const double* input,
    unsigned char* output,
    volatile unsigned int* const __restrict__ cmpOffset, //translated comment
    volatile unsigned int* const __restrict__ locOffset, //translated comment
    volatile int* const __restrict__ flag,             //， warp （
    int totalSize
)
{
    //translated comment
    __shared__ unsigned int excl_sum; //translated comment
    //__shared__ unsigned int base_idx; // warp

    //translated comment
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;                   //warp （0-31）
    const int warp = idx >> 5;                     //warp

    //translated comment
    int startIdx = idx * DATA_PER_THREAD;
    int endIdx = min(startIdx + DATA_PER_THREAD, totalSize);
    int numDatas = max(0,endIdx - startIdx);
    
    uint64_t deltas[DATA_PER_THREAD]={0};

    if((idx-1)*DATA_PER_THREAD>totalSize){
        return;
    }
    int maxDecimalPlaces = 0;
    long firstValue = 0;
    volatile int bitCount = 0;
    volatile int maxBeta = 0;

    int base_block_start_idx=0;
    // int base_block_end_idx=0;
    // int quant_chunk_idx;
    //int block_idx; // ，

    long currQuant=0;
    long lorenQuant=0;
    long prevQuant=0;

    unsigned int thread_ofs = 0;

    int maxSp = -99;
    //translated comment
    for (int i = 0; i < numDatas; i++) {
        double value =input[startIdx + i];
        double log10v = log10(std::abs(value));
        int sp = floor(log10v);
        maxSp = device_max(maxSp, sp);
        double alpha = getDecimalPlaces(value, sp);//translated comment
        // double beta =  alpha + sp + 1;
        // maxBeta = device_max(maxBeta,beta);
        // if(alpha>maxDecimalPlaces){
        //     idwrong = i;
        // }
        maxDecimalPlaces = device_max(maxDecimalPlaces, alpha);
    }

    maxBeta = maxSp + maxDecimalPlaces+1;
    
    // for (int i = 0; i < numDatas; i++) {
    //     double value =input[startIdx + i];
    //     if(value == -4.0023584){
    //     printf("maxBeta:%d, maxAlpha: %d, maxSp: %d, wrongAlpha:%.16f \n", maxBeta, maxDecimalPlaces, maxSp,input[startIdx + idwrong]);
    // }
    // }
    //printf("maxDecimalPlaces:%d\n", maxDecimalPlaces);
    //2. FOR + zigzag（ 4 ）
    volatile uint64_t maxDelta = 0;
    firstValue = double2long(input[startIdx], maxDecimalPlaces,maxBeta); //translated comment
    prevQuant = firstValue;//translated comment
    base_block_start_idx = startIdx + 1;

    for(int i=0;i<numDatas-1;i++){
        currQuant = double2long(input[base_block_start_idx+i], maxDecimalPlaces,maxBeta); //translated comment
        lorenQuant = currQuant - prevQuant; //translated comment
        deltas[i] = zigzag_encode_cuda(lorenQuant);
    
        maxDelta = device_max_uint64(maxDelta, deltas[i]);
        prevQuant = currQuant;

    }

    bitCount = maxDelta > 0 ? 64 - __clzll(maxDelta) : 1;//translated comment
    bitCount = min(bitCount, (int)MAX_BITCOUNT);

        const int numByte = (numDatas - 1 + 7) / 8;
        uint8_t result_flat[8192] = {};

        for (int i = 0; i < bitCount; ++i) {//translated comment
            int j=0;
            while((j+8+1)<numDatas)
            {
                int byteIndex = j / 8;  //bit
                uint8_t currentByte = 0;
                currentByte |= (((deltas[j] >> (bitCount - 1 - i)) & 1) << 7);
                currentByte |= (((deltas[j+1] >> (bitCount - 1 - i)) & 1) << 6);
                currentByte |= (((deltas[j+2] >> (bitCount - 1 - i)) & 1) << 5);
                currentByte |= (((deltas[j+3] >> (bitCount - 1 - i)) & 1) << 4);
                currentByte |= (((deltas[j+4] >> (bitCount - 1 - i)) & 1) << 3);
                currentByte |= (((deltas[j+5] >> (bitCount - 1 - i)) & 1) << 2);
                currentByte |= (((deltas[j+6] >> (bitCount - 1 - i)) & 1) << 1);
                currentByte |= (((deltas[j+7] >> (bitCount - 1 - i)) & 1) << 0);
                
                result_flat[i * numByte + byteIndex] = currentByte;
                j+=8;
            }
            for (; j <(numDatas -1); ++j) {//numBytes

                int byteIndex = j / 8;  //bit
                int bitIndex = j % 8;   //bit

                uint8_t bitVal = ((deltas[j] >> (bitCount - 1 - i)) & 1);
                //bit


                result_flat[i * numByte + byteIndex] |= bitVal << (7 - bitIndex);


            }

        }


        //4.2 ， ， bitsize
        uint64_t bitSize =  64ULL +                 // bitsize
                            64ULL +                 // firstValue
                            8ULL +                  // maxDecimalPlaces
                            8ULL +                  // maxBeta
                            8ULL +                  // bitCount
                            64ULL;                  // flag1

        uint64_t flag1 = 0;
        uint8_t flag2[(DATA_PER_THREAD-1)];

        memset(flag2,0,sizeof(flag2));

        for(int i = 0;i<bitCount;i++){
            int b0 = 0;
            int b1 = 0;

            //result_flat flag2_flat
            size_t result_row_start_offset = i * numByte;
            size_t flag2_row_start_offset = i * ((numByte + 7) / 8); //flag2
        
            for (int j = 0; j < numByte; j++) {
                uint8_t current_result_byte = result_flat[result_row_start_offset + j];

                b0 += (current_result_byte == 0);
                b1 += (current_result_byte != 0);

                int flag2_byte_idx = j / 8;
                int flag2_bit_idx = j % 8;

                // uint8_t mask = (1 << flag2_bit_idx);
                //flag2 ，
                if (current_result_byte != 0) {
                    flag2[flag2_row_start_offset + flag2_byte_idx]|= (current_result_byte != 0) << flag2_bit_idx;
                } else {
                    flag2[flag2_row_start_offset + flag2_byte_idx]&= ~((current_result_byte == 0) << flag2_bit_idx);
                }
            }
            uint64_t is_sparse = 1;//(uint64_t)(((numByte + 7) / 8 + b1) < numByte);
                // flag1|=(1<< i);
                flag1 |= (is_sparse << i);
                flag1 &= ~((!is_sparse) << i);
                bitSize += is_sparse ? ((numByte + 7) / 8 + b1) * 8 : 8 * numByte;
        }


        if(numDatas<=0)
        {
            bitSize=0;
        }
    //translated comment
        thread_ofs+=bitSize;//bitSize bit

        //5.1. Warp( ) ，
        #pragma unroll 5
        for(int i = 1; i < 32; i <<= 1)
        {
            int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
            if(lane >= i) thread_ofs += tmp;                      //translated comment
        }
        __syncthreads(); //translated comment
        // printf("thread_ofs[%d]:%d",lane,thread_ofs);

        //5.2 Warp( ) locOffset flag
        if(lane == 31||numDatas<=0)//translated comment
        {
            locOffset[warp + 1] = thread_ofs; //warp
            __threadfence();                  //translated comment
            if(warp == 0)
            {
                flag[0] = 2;                   //warp
                __threadfence();
                flag[1] = 1;                   //warp
                __threadfence();
            }
            else
            {
                flag[warp + 1] = 1;            //warp
                __threadfence();
            }
            //printf("flag[%d] ready\n",warp + 1);
        }
        __syncthreads(); //， flag

        //5.3 warp， （ ）
        if(warp > 0)
        {
            if(!lane) //warp
            {
                int lookback = warp;          //warp( )
                int loc_excl_sum = 0;         //translated comment

                while(lookback > 0)//wrap（ ）
                {
                    int status;
                    do{
                        status = flag[lookback]; //warp
                    //    printf(" loop flag[%d]:%d\n",lookback,status);
                        __threadfence();         //translated comment
                    } while(status == 0);

                    if(status == 2)
                    {
                        loc_excl_sum += cmpOffset[lookback]; //warp cmpOffset
                        __threadfence();
                        break;
                    }
                    if(status == 1)
                        loc_excl_sum += locOffset[lookback]; //warp locOffset
                    lookback--;
                    __threadfence();
                   // printf(" turn flag[%d]:%d\n",lookback,status);
                }
                //printf(" loop out warp:%d\n",warp);
                excl_sum = loc_excl_sum; //translated comment

                cmpOffset[warp] = excl_sum; //warp cmpOffset
                __threadfence();           //translated comment

                //printf("flag[%d] over1\n",warp);
                if(warp == gridDim.x - 1)
                {
                    cmpOffset[warp + 1] = cmpOffset[warp] + locOffset[warp + 1]; //warp cmpOffset
                    __threadfence();

                }
                flag[warp] = 2;             //warp
                //printf("flag[%d] over2\n",warp);
                __threadfence();
            }
        }
        else {
            //warp==0： 0（ ，block ）
            if (!lane) { excl_sum = 0; }
        }
        __syncthreads(); //， cmpOffset
        if(numDatas<=0)
        {
            if(cmpOffset[warp + 1]<=0)
            {
                cmpOffset[warp + 1] = cmpOffset[warp] + locOffset[warp + 1];
            }
            return;
        }
        //translated comment
        int outputIdxBit = excl_sum + thread_ofs - bitSize; //bit wrap +wrap
        int outputIdx = (outputIdxBit+7)/8;

    //translated comment

    


        // unsigned long long firstValueBits = 0;
        // memcpy(&firstValueBits, &firstValue, sizeof(long));
        // if (outputIdx % 8 != 0) {
        //6.1 bitSize (8 )
            // for(int i = 0; i < 8; i++) {
            //     output[outputIdx + i] = (bitSize >> (i * 8)) & 0xFF;

            // }


            //// 6.2. firstValue (8 )
            // for(int i = 0; i < 8; i++) {
            //     output[outputIdx + 8 + i] = (firstValueBits >> (i * 8)) & 0xFF;

            // }
        memcpy(output + outputIdx, &bitSize, sizeof(unsigned long long));

        //6.2. firstValue (8 )
        //firstValue (double) output
        memcpy(output + outputIdx + 8, &firstValue, sizeof(double)); //sizeof(double)
            

        // }
        //6.3. maxDecimalPlaces bitCount ( 1 )
        output[outputIdx + 16] = static_cast<unsigned char>(maxDecimalPlaces);
        output[outputIdx + 17] = static_cast<unsigned char>(maxBeta);
        output[outputIdx + 18] = static_cast<unsigned char>(bitCount);

        //6.4 flag1(8 )
        // for(int i = 0; i < 8; i++) {
        //     output[outputIdx + 19 + i] = (flag1 >> (i * 8)) & 0xFF;
        // }

        memcpy(output + outputIdx + 19, &flag1, sizeof(unsigned long long));
        // printf("In %d  flag1 is : %llx\n",idx,flag1);
        //translated comment
        int flag2Byte = (numByte+7)/8;
        int ofs=outputIdx + 27;
        //int res=0; //byte bit
        for(int i=0;i<bitCount;i++)
        {
            size_t flag2_row_start_offset = i * flag2Byte;
            size_t result_row_start_offset = i * numByte;
            if((flag1 & (1ULL << i)) != 0){
                memcpy(output + ofs, flag2 + flag2_row_start_offset, flag2Byte);
                ofs += flag2Byte;
                for (int j = 0; j < numByte; j++) {
                    if (result_flat[result_row_start_offset + j]) {
                        output[ofs++] = result_flat[result_row_start_offset + j];
                    }
                }
            } else { //translated comment
                memcpy(output + ofs, result_flat + result_row_start_offset, numByte);
                ofs += numByte;
            }

                
        }

}



void FalconCompressor::Falcon_compress_spare(double* d_oriData, unsigned char* d_cmpBytes, unsigned int* d2h_async_totalBits_ptr, size_t nbEle, cudaStream_t stream)
{
    // Data blocking.
    int bsize = cmp_tblock_size;
    int gsize = (nbEle + bsize * DATA_PER_THREAD - 1) / (bsize * DATA_PER_THREAD);
    // size_t numthread = (nbEle + cmp_chunk - 1) / (cmp_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    cudaMallocAsync((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMemsetAsync(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMallocAsync((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMemsetAsync(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMallocAsync((void**)&d_flag, sizeof(int)*cmpOffSize,stream);
    cudaMemsetAsync(d_flag, 0, sizeof(int)*cmpOffSize,stream);


    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);

    Falcon_compress_kernel_spare<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, nbEle);

    // Obtain compression ratio and move data back to CPU.  
    cudaMemcpyAsync(d2h_async_totalBits_ptr, (d_cmpOffset + cmpOffSize-1), sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaFreeAsync(d_cmpOffset,stream);
    cudaFreeAsync(d_locOffset,stream);
    cudaFreeAsync(d_flag,stream);
}

/*

__global__ void Falcon_compress_kernel_string(
    const double* input,
    unsigned char* output,
    volatile unsigned int* const __restrict__ cmpOffset, //translated comment
    volatile unsigned int* const __restrict__ locOffset, //translated comment
    volatile int* const __restrict__ flag,             //， warp （
    int totalSize
)
{
    //translated comment
    __shared__ unsigned int excl_sum; //translated comment
    //__shared__ unsigned int base_idx; // warp

    //translated comment
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;                   //warp （0-31）
    const int warp = idx >> 5;                     //warp

    //translated comment
    int startIdx = idx * DATA_PER_THREAD;
    int endIdx = min(startIdx + DATA_PER_THREAD, totalSize);
    int numDatas = endIdx - startIdx;
    uint64_t deltas[DATA_PER_THREAD]={};

    int maxDecimalPlaces = 0;
    int maxBeta =0;
    long firstValue = 0;
    int bitCount = 0;

    int base_block_start_idx=0;
    // int base_block_end_idx=0;
    // int quant_chunk_idx;
    //int block_idx; // ，

    long currQuant=0;
    long lorenQuant=0;
    long prevQuant=0;

    unsigned int thread_ofs = 0;
    //translated comment
    for (int i = 0; i < numDatas; i++) {
        double value =input[startIdx + i];
        double log10v = log10(std::abs(value));
        int sp = floor(log10v);
        
        double alpha = getDecimalPlaces_string(value, sp);//translated comment
        double beta =  alpha + sp + 1;
        maxBeta = device_max(maxBeta,beta);
        maxDecimalPlaces = device_max(maxDecimalPlaces, alpha);
    }
    //printf("maxDecimalPlaces:%d\n", maxDecimalPlaces);
    //2. FOR + zigzag（ 4 ）
    uint64_t maxDelta = 0;
    firstValue = double2long(input[startIdx], maxDecimalPlaces,maxBeta); //translated comment
    prevQuant = firstValue;//translated comment
    base_block_start_idx = startIdx + 1;

    for(int i=0;i<numDatas-1;i++){
        currQuant = double2long(input[base_block_start_idx+i], maxDecimalPlaces,maxBeta); //translated comment
        lorenQuant = currQuant - prevQuant; //translated comment
        deltas[i] = zigzag_encode_cuda(lorenQuant);
    
        maxDelta = device_max_uint64(maxDelta, deltas[i]);
        prevQuant = currQuant;
        // if(startIdx<4){
        //     printf("input %d, ",deltas[i]);
        // }
    }
    //for(int j = 0; j < (numDatas+31) / 32; j++) { // / （32）
    //base_block_start_idx = startIdx + j * 32; // 32
    //base_block_end_idx = base_block_start_idx + 32; // 32

    //     if(base_block_end_idx < totalSize) {
    //             int i = base_block_start_idx;
                
            
    //#pragma unroll 8 // 8 ， 4*8=32 , 7 ，
    //         for(; i < base_block_end_idx; i += 4) {

    //             tmp_buffer = reinterpret_cast<const double4*>(input)[(i) / 4];
    //quant_chunk_idx = j * 32 + ((i) % 32); //

    //currQuant = double2long(tmp_buffer.x, maxDecimalPlaces,maxBeta); //
    //lorenQuant = currQuant - prevQuant; //

    //             deltas[quant_chunk_idx] = zigzag_encode_cuda(lorenQuant);
    //             //printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx],quant_chunk_idx,lorenQuant,currQuant,prevQuant);
    //prevQuant = currQuant; //
    //maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx]); //
    //             // }

    //translated comment
    //             currQuant = double2long(tmp_buffer.y, maxDecimalPlaces,maxBeta);
    //             lorenQuant = currQuant - prevQuant;

    //             deltas[quant_chunk_idx + 1] = zigzag_encode_cuda(lorenQuant);
    //             //printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx+1],quant_chunk_idx+1,lorenQuant,currQuant,prevQuant);
    //             prevQuant = currQuant;
    //             maxDelta = device_max_uint64(maxDelta, deltas[ quant_chunk_idx + 1]);

    //translated comment
    //             currQuant = double2long(tmp_buffer.z, maxDecimalPlaces,maxBeta);
    //             lorenQuant = currQuant - prevQuant;

    //             deltas[quant_chunk_idx + 2] = zigzag_encode_cuda(lorenQuant);
    //             // printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx+2],quant_chunk_idx+2,lorenQuant,currQuant,prevQuant);
    //             prevQuant = currQuant;
    //             maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx + 2]);

    //translated comment
    //             currQuant = double2long(tmp_buffer.w, maxDecimalPlaces,maxBeta);
    //             lorenQuant = currQuant - prevQuant;

    //             deltas[quant_chunk_idx + 3] = zigzag_encode_cuda(lorenQuant);
    //             // printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx+3],quant_chunk_idx+3,lorenQuant,currQuant,prevQuant);
    //             prevQuant = currQuant;
    //             maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx + 3]);
    //         }
    //     }
    //     else {
    //translated comment
    //         if(base_block_start_idx >= endIdx) {
    //// ， absQuant 0
    //             quant_chunk_idx = j * 32 + (base_block_start_idx % 32);
    //             for(int i = quant_chunk_idx; i < quant_chunk_idx + 32; i++)
    //                 deltas[i] = 0;
    //         }
    //         else {
    //translated comment
    //int remainbEle = totalSize - base_block_start_idx; //
    //int zeronbEle = base_block_end_idx - totalSize; //

    //translated comment
    //             for(int i = base_block_start_idx; i < base_block_start_idx + remainbEle; i++) {
    //                 if(i==startIdx)
    //                 {
    //                     deltas[0]=0;
    //                     continue;
    //                 }
    //                 quant_chunk_idx = j * 32 + (i % 32);
    //                 currQuant = double2long(input[i], maxDecimalPlaces,maxBeta);

    //                 lorenQuant = currQuant - prevQuant;

    //                 deltas[ quant_chunk_idx] = zigzag_encode_cuda(lorenQuant);

    //                 prevQuant = currQuant;
    //                 maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx]);
    //             }

    //             quant_chunk_idx = j * 32 + (totalSize % 32);
    //             for(int i = quant_chunk_idx; i < quant_chunk_idx + zeronbEle; i++)
    //                 deltas[i] = 0;
    //         }
    //     }
    // }

    bitCount = maxDelta > 0 ? 64 - __clzll(maxDelta) : 1;//translated comment
    bitCount = min(bitCount, (int)MAX_BITCOUNT);


        int numByte = (numDatas-1 + 7) / 8;
        // uint8_t result[64][128];
        uint8_t result[64][128] = {}; 
        //translated comment

        //uint64_t
        for (int i = 0; i < bitCount; ++i) {//translated comment
            int j=0;

            while(j+8<numDatas-1)//translated comment
            {
                int byteIndex = j / 8;  //bit
                result[i][byteIndex] = result[i][byteIndex] |
                                        (((deltas[j] >> (bitCount - 1 - i)) & 1) << (7))|
                                        (((deltas[j+1] >> (bitCount - 1 - i)) & 1) << (6))|
                                        (((deltas[j+2] >> (bitCount - 1 - i)) & 1) << (5))|
                                        (((deltas[j+3] >> (bitCount - 1 - i)) & 1) << (4))|
                                        (((deltas[j+4] >> (bitCount - 1 - i)) & 1) << (3))|
                                        (((deltas[j+5] >> (bitCount - 1 - i)) & 1) << (2))|
                                        (((deltas[j+6] >> (bitCount - 1 - i)) & 1) << (1))|
                                        (((deltas[j+7] >> (bitCount - 1 - i)) & 1) << (0));
                j+=8;
            }
            for (; j <numDatas -1; ++j) {//numBytes
                //（ bit ）
                // if(i==0)
                // {
                //printf("0x %02x ", deltas[j]); // ， 2
                // }
                int byteIndex = j / 8;  //bit
                int bitIndex = j % 8;   //bit

                //bit
                result[i][byteIndex] |= (((deltas[j] >> (bitCount - 1 - i)) & 1) << (7 - bitIndex));
            }

        }

        //4.2 ， ， bitsize
        uint64_t bitSize =  64ULL +                 // bitsize
                            64ULL +                 // firstValue
                            8ULL +                  // maxDecimalPlaces
                            8ULL +                  // maxBeta
                            8ULL +                  // bitCount
                            64ULL;                  // flag1

        uint64_t flag1 = 0;              //translated comment
        uint8_t flag2[64][16];          //, 1024 ， 1024bit， 128byte,
        memset(flag2, 0, sizeof(flag2));

        int BITS_PER_THREAD=4;
        for(int i = 0; i < bitCount; i += BITS_PER_THREAD) { //translated comment
            for(int b = 0; b < BITS_PER_THREAD && (i + b) < bitCount; ++b) {
                int bit = i + b;
                int b0 = 0;
                int b1 = 0;
                for(int j = 0; j < numByte; j++) {
                    int m_byte = j / 8;
                    int m_bit = j % 8;
                    uint8_t current_result = result[bit][j];
                    b0 += (current_result == 0);
                    b1 += (current_result != 0);
                    flag2[bit][m_byte] |= (current_result != 0) << m_bit;//translated comment
                    flag2[bit][m_byte] &= ~((current_result == 0) << m_bit);//translated comment
                }
                //translated comment
                uint64_t is_sparse = ((numByte + 7) / 8 + b1) < numByte;
                flag1 |= (is_sparse << bit);
                flag1 &= ~((!is_sparse) << bit);
                bitSize += is_sparse ? ((numByte + 7) / 8 + b1) * 8 : 8 * numByte;
                //flag2 +b1 numByte*8
            }
        }


        if(numDatas<=0)
        {
            bitSize=0;
        }
    //translated comment
        thread_ofs+=bitSize;//bitSize bit

        //5.1. Warp( ) ，
        #pragma unroll 5
        for(int i = 1; i < 32; i <<= 1)
        {
            int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
            if(lane >= i) thread_ofs += tmp;                      //translated comment
        }
        __syncthreads(); //translated comment
        // printf("thread_ofs[%d]:%d",lane,thread_ofs);

        //5.2 Warp( ) locOffset flag
        if(lane == 31||numDatas<=0)//translated comment
        {
            locOffset[warp + 1] = thread_ofs; //warp
            __threadfence();                  //translated comment
            if(warp == 0)
            {
                flag[0] = 2;                   //warp
                __threadfence();
                flag[1] = 1;                   //warp
                __threadfence();
            }
            else
            {
                flag[warp + 1] = 1;            //warp
                __threadfence();
            }
            //printf("flag[%d] ready\n",warp + 1);
        }
        __syncthreads(); //， flag

        //5.3 warp， （ ）
        if(warp > 0)
        {
            if(!lane) //warp
            {
                int lookback = warp;          //warp( )
                int loc_excl_sum = 0;         //translated comment

                while(lookback > 0)//wrap（ ）
                {
                    int status;
                    do{
                        status = flag[lookback]; //warp
                    //    printf(" loop flag[%d]:%d\n",lookback,status);
                        __threadfence();         //translated comment
                    } while(status == 0);

                    if(status == 2)
                    {
                        loc_excl_sum += cmpOffset[lookback]; //warp cmpOffset
                        __threadfence();
                        break;
                    }
                    if(status == 1)
                        loc_excl_sum += locOffset[lookback]; //warp locOffset
                    lookback--;
                    __threadfence();
                   // printf(" turn flag[%d]:%d\n",lookback,status);
                }
                //printf(" loop out warp:%d\n",warp);
                excl_sum = loc_excl_sum; //translated comment

                cmpOffset[warp] = excl_sum; //warp cmpOffset
                __threadfence();           //translated comment

                //printf("flag[%d] over1\n",warp);
                if(warp == gridDim.x - 1)
                {
                    cmpOffset[warp + 1] = cmpOffset[warp] + locOffset[warp + 1]; //warp cmpOffset
                    __threadfence();

                }
                flag[warp] = 2;             //warp
                //printf("flag[%d] over2\n",warp);
                __threadfence();
            }
        }
        else {
            //warp==0： 0（ ，block ）
            if (!lane) { excl_sum = 0; }
        }
        __syncthreads(); //， cmpOffset
        if(numDatas<=0)
        {
            if(cmpOffset[warp + 1]<=0)
            {
                cmpOffset[warp + 1] = cmpOffset[warp] + locOffset[warp + 1];
            }
            return;
        }
        //translated comment
        int outputIdxBit = excl_sum + thread_ofs - bitSize; //bit wrap +wrap
        int outputIdx = (outputIdxBit+7)/8;

    //translated comment


        unsigned long long firstValueBits = 0;
        memcpy(&firstValueBits, &firstValue, sizeof(long));
        // if (outputIdx % 8 != 0) {
        //6.1 bitSize (8 )
            for(int i = 0; i < 8; i++) {
                output[outputIdx + i] = (bitSize >> (i * 8)) & 0xFF;

            }


            //6.2. firstValue (8 )
            for(int i = 0; i < 8; i++) {
                output[outputIdx + 8 + i] = (firstValueBits >> (i * 8)) & 0xFF;

            }

        // }
        //6.3. maxDecimalPlaces bitCount ( 1 )
        output[outputIdx + 16] = static_cast<unsigned char>(maxDecimalPlaces);
        output[outputIdx + 17] = static_cast<unsigned char>(maxBeta);
        output[outputIdx + 18] = static_cast<unsigned char>(bitCount);

        //6.4 flag1(8 )
        for(int i = 0; i < 8; i++) {
            output[outputIdx + 19 + i] = (flag1 >> (i * 8)) & 0xFF;
        }
        // printf("In %d  flag1 is : %llx\n",idx,flag1);
        //translated comment
        int flag2Byte=(numByte+7)/8;
        int ofs=outputIdx + 27;
        //int res=0; //byte bit
        for(int i=0;i<bitCount;i++)
        {
            if((flag1 & (1ULL << i)) != 0)//flag i bit 0:
            {
                //6.5.1 flag2+data
                for(int j=0;j<flag2Byte;j++)
                {
                    output[ofs++] = static_cast<unsigned char>(flag2[i][j]);
                    // printf("flag2[%d][%d]:0x%llx\n",i,j,flag2[i][j]);
                }
                for(int j=0;j<numByte;j++)
                {
                    if(result[i][j])
                    {
                        output[ofs++] = static_cast<unsigned char>(result[i][j]);
                    }
                }
            }
            else{
                //6.5.2 data

                for(int j=0;j<numByte;j++)
                {
                    output[ofs++] = static_cast<unsigned char>(result[i][j]);
                }
            }

        }


}

//bits
void FalconCompressor::Falcon_compress_string(double* d_oriData, unsigned char* d_cmpBytes, unsigned int* d2h_async_totalBits_ptr, size_t nbEle, cudaStream_t stream)
{
    // Data blocking.
    int bsize = cmp_tblock_size;
    int gsize = (nbEle + bsize * DATA_PER_THREAD - 1) / (bsize * DATA_PER_THREAD);
    // size_t numthread = (nbEle + cmp_chunk - 1) / (cmp_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    cudaMallocAsync((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMemsetAsync(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMallocAsync((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMemsetAsync(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMallocAsync((void**)&d_flag, sizeof(int)*cmpOffSize,stream);
    cudaMemsetAsync(d_flag, 0, sizeof(int)*cmpOffSize,stream);


    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);

    Falcon_compress_kernel_string<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, nbEle);

    // Obtain compression ratio and move data back to CPU.  
    cudaMemcpyAsync(d2h_async_totalBits_ptr, (d_cmpOffset + cmpOffSize-1), sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaFreeAsync(d_cmpOffset,stream);
    cudaFreeAsync(d_locOffset,stream);
    cudaFreeAsync(d_flag,stream);
}
*/
