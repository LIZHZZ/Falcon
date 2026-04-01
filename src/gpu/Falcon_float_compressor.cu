//
#include "Falcon_float_compressor.cuh"
#include <iomanip> // For formatted debug output
// Constant definitions

// pow10_table and POW_NUM_G
__constant__ float pow10_table[8] = {
    1.0,        // 10^0
    10.0,       // 10^1
    100.0,      // 10^2
    1000.0,     // 10^3
    10000.0,    // 10^4
    100000.0,   // 10^5
    1000000.0,  // 10^6
    10000000.0, // 10^7
};

// ZigZag encoding helper
// __device__ static uint64_t zigzag_encode_cuda(int64_t value) {
//     return (value << 1) ^ (value >> 63);
// }
__device__ __forceinline__ static unsigned int zigzag_encode_cuda(int value)
{
    return (value << 1) ^ (value >> 31);
}

__device__ static int getDecimalPlaces(float value, int sp)
{
    float trac = value + POW_NUM_G - POW_NUM_G;
    float temp = value;

    int digits = 0;
    float td = 1;
    float deltaBound = abs(value) * pow(2, -23);
    while (abs(temp - trac) >= deltaBound * td && digits < 8 - sp - 1)
    {
        digits++;
        td = pow10_table[digits];
        temp = value * td;
        trac = temp + POW_NUM_G - POW_NUM_G;
    }
    if(round(temp)/td!=value)
    {
        digits=17;
        
    }
    return digits;
}
/*
    __device__ static int getDecimalPlaces_s(float v, int sp)
    {
        //     value = value < 0 ? -value : value;
        //     double trac = value + POW_NUM_G - POW_NUM_G;
        //     double temp = value;

        //     int digits = 0;
        //     double td = 1;
        //     double deltaBound = abs(value) * pow(2, -52);
        //     while (abs(temp - trac) == 0 && digits < 16 - sp - 1)
        //     {
        //         digits++;
        //         td = pow10_table[digits];
        //         temp = value * td;
        //         trac = temp + POW_NUM_G - POW_NUM_G;
        //     }

        //     return digits;
        // }
        v = v < 0 ? -v : v;

        int i = 0;
        float scale = 1.0;

        // Find the smallest multiplier that turns v into an exact integer
        while (i < 9)
        {
            float temp = v * scale;

            if (round(temp) == temp)
            {

                return i;
            }
            i++;
            scale *= 10.0;
        }
        return 9; // Reached single-precision limit
    }
*/
// Helper: print a specified range of the buffer in hexadecimal
__device__ void print_bytes(const unsigned char *buffer, size_t start, size_t length, const char *label)
{
    printf("%s: ", label);
    for (size_t i = start; i < start + length; ++i)
    {
        // Print each byte in hexadecimal form
        printf("%02x ", buffer[i]);
    }
    printf("\n");
}

#define cudaCheckError(ans)                   \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__device__ inline int device_min(int a, int b)
{
    return (a < b) ? a : b;
}

__device__ inline int device_max(int a, int b)
{
    return (a > b) ? a : b;
}

__device__ inline uint32_t device_min_uint32(uint32_t a, uint32_t b)
{
    return (a < b) ? a : b;
}

__device__ inline uint32_t device_max_uint32(uint32_t a, uint32_t b)
{
    return (a > b) ? a : b;
}
__device__ int encodeFloatWithSignLast(float x)
{
    union
    {
        float d;
        int u;
    } val;

    val.d = x;

    return (val.u << 1) ^ (val.u >> (sizeof(int) * 8 - 1));
}

__device__ inline int float2int(float data, int maxDecimalPlaces, int maxBeta)
{

    return (maxBeta > 6)
               ? (encodeFloatWithSignLast(data))
               : static_cast<int>(round(data * pow10_table[maxDecimalPlaces]));
}


__global__ void Falcon_compress_kernel(
    const float *input,
    unsigned char *output,
    volatile unsigned int *const __restrict__ cmpOffset, // Compressed data offset array (output)
    volatile unsigned int *const __restrict__ locOffset, // Local offset array (output)
    volatile int *const __restrict__ flag,               // Flag array for synchronizing warp states (output)
    int totalSize)
{
    // Shared memory used within the block
    __shared__ unsigned int excl_sum; // Exclusive prefix sum used for offset computation
    //__shared__ unsigned int base_idx; // Base index of the current warp (unused)

    // Thread and block indices
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f; // Lane index within the warp (0-31)
    const int warp = idx >> 5;   // Warp index

    // Each thread processes DATA_PER_THREAD elements
    int startIdx = idx * DATA_PER_THREAD;
    int endIdx = min(startIdx + DATA_PER_THREAD, totalSize);
    int numDatas = endIdx - startIdx;
    uint32_t deltas[DATA_PER_THREAD] = {};

    int maxDecimalPlaces = 0;
    int maxBeta = 0;
    int firstValue = 0;
    int bitCount = 0;

    int base_block_start_idx;// base_block_end_idx;
    // int quant_chunk_idx;
    // int block_idx; // Can be removed if unused

    int currQuant = 0;
    int lorenQuant = 0;
    int prevQuant = 0;

    unsigned int thread_ofs = 0;
    // float4 tmp_buffer;
    // float maxDeV = 0;
    int maxSp = -99;
    // 1. Sampling
    for (int i = 0; i < numDatas; i++)
    {
        float value = input[startIdx + i];
        float log10v = log10(std::abs(value));
        int sp = floor(log10v);
        maxSp = device_max(maxSp, sp);  
        float alpha = getDecimalPlaces(value, sp); // Number of decimal places
        // float beta = alpha + sp + 1;
        // maxBeta = device_max(maxBeta, beta);
        // if (maxDecimalPlaces < alpha)
        // {
        //     maxDeV = value;
        // }
        maxDecimalPlaces = device_max(maxDecimalPlaces, alpha);
    }
    maxBeta = maxSp + maxDecimalPlaces+1;
    //  2. FOR + zigzag (vectorized)
    uint32_t maxDelta = 0;
    firstValue = float2int(input[startIdx], maxDecimalPlaces, maxBeta); // Quantize the first value
    prevQuant = firstValue;                                             // Initialize previous quantized value
        
        base_block_start_idx = startIdx + 1;

        for(int i=0;i<numDatas-1;i++){
            currQuant = float2int(input[base_block_start_idx+i], maxDecimalPlaces,maxBeta); // Quantize current point
            lorenQuant = currQuant - prevQuant; // Compute delta
            deltas[i] = zigzag_encode_cuda(lorenQuant);
        
            maxDelta = device_max_uint32(maxDelta, deltas[i]);
            prevQuant = currQuant;
            // if(startIdx<4){
            //     printf("input %d, ",deltas[i]);
            // }
        }
    bitCount = maxDelta > 0 ? 32 - __clz(maxDelta) : 1; // Use intrinsic instead of manual loop
    bitCount = min(bitCount, (int)MAX_BITCOUNT);

    int numByte = (numDatas-1 + 7) / 8;
    // uint8_t result[64][128];
    uint8_t result[32][128] = {};
    // Initialize 2D array and iterate over each bit-plane
    for (int i = 0; i < bitCount; ++i)
    { // Row
        int j = 0;

        while (j + 8 < numDatas-1) // Slight optimization: process 8 bits at a time
        {
            int byteIndex = j / 8; // Which byte the current bit belongs to
            result[i][byteIndex] = result[i][byteIndex] |
                                   (((deltas[j] >> (bitCount - 1 - i)) & 1) << (7)) |
                                   (((deltas[j + 1] >> (bitCount - 1 - i)) & 1) << (6)) |
                                   (((deltas[j + 2] >> (bitCount - 1 - i)) & 1) << (5)) |
                                   (((deltas[j + 3] >> (bitCount - 1 - i)) & 1) << (4)) |
                                   (((deltas[j + 4] >> (bitCount - 1 - i)) & 1) << (3)) |
                                   (((deltas[j + 5] >> (bitCount - 1 - i)) & 1) << (2)) |
                                   (((deltas[j + 6] >> (bitCount - 1 - i)) & 1) << (1)) |
                                   (((deltas[j + 7] >> (bitCount - 1 - i)) & 1) << (0));
            j += 8;
        }
        for (; j < numDatas-1; ++j)
        { // Column (numBytes)
            int byteIndex = j / 8; // Which byte the current bit belongs to
            int bitIndex = j % 8;  // Bit position within the byte

            // Extract the current bit and store it in the result array
            result[i][byteIndex] |= (((deltas[j] >> (bitCount - 1 - i)) & 1) << (7 - bitIndex));
        }
    }

    // 4.2 Mark sparse columns and compute bitSize
    uint64_t bitSize = 32 +   // bitsize
                       32 +   // firstValue
                       8ULL + // maxDecimalPlaces
                       8ULL + // maxBeta
                       8ULL + // bitCount
                       32;    // flag1

    uint32_t flag1 = 0;    // Records whether each column is sparse
    uint8_t flag2[32][16]; // For sparse columns, records sparse positions (up to 1024 bits -> 128 bytes)
    memset(flag2, 0, sizeof(flag2));

    // for(int i=0;i<1024;i++){
    //     if(startIdx + i ==23704){
    //         printf("tid: %d,  maxDecimal:%d  maxBeta: %d\n",idx,maxDecimalPlaces,maxBeta);
    //         printf("value: %8f, errorDeV: %8f \n",input[startIdx + i], maxDeV);
    //     }
    // }

    int BITS_PER_THREAD = 4;
    for (int i = 0; i < bitCount; i += BITS_PER_THREAD)
    { // Process BITS_PER_THREAD bit-planes at a time
        for (int b = 0; b < BITS_PER_THREAD && (i + b) < bitCount; ++b)
        {
            int bit = i + b;
            int b0 = 0;
            int b1 = 0;
            for (int j = 0; j < numByte; j++)
            {
                int m_byte = j / 8;
                int m_bit = j % 8;
                uint8_t current_result = result[bit][j];
                b0 += (current_result == 0);
                b1 += (current_result != 0);
                flag2[bit][m_byte] |= (current_result != 0) << m_bit;    // Set bit
                flag2[bit][m_byte] &= ~((current_result == 0) << m_bit); // Clear bit
            }
            // Use mask and arithmetic instead of branches (small optimization)
            uint32_t is_sparse = ((numByte + 7) / 8 + b1) < numByte;
            flag1 |= (is_sparse << bit);
            flag1 &= ~((!is_sparse) << bit);
            bitSize += is_sparse ? ((numByte + 7) / 8 + b1) * 8 : 8 * numByte;
            // flag2 length + b1, or numByte * 8
        }
    }

    if (numDatas <= 0)
    {
        bitSize = 0;
    }
    // 5. Prefix-sum computation
    thread_ofs += bitSize; // bitSize is the number of bits each thread will write

// 5.1. Warp-level prefix sum to determine per-thread byte offsets
#pragma unroll 5
    for (int i = 1; i < 32; i <<= 1)
    {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if (lane >= i)
            thread_ofs += tmp; // Accumulate offset
    }
    __syncthreads(); // Ensure prefix sum is complete

    // 5.2 Last warp lane updates locOffset and flag arrays
    if (lane == 31 || numDatas <= 0) // Or last active thread when fewer than 32 threads
    {
        locOffset[warp + 1] = thread_ofs; // Update local offset for the next warp
        __threadfence();                  // Ensure global write has completed
        if (warp == 0)
        {
            flag[0] = 2; // Mark first warp as having completed prefix sum
            __threadfence();
            flag[1] = 1; // Mark next warp as ready to start
            __threadfence();
        }
        else
        {
            flag[warp + 1] = 1; // Mark next warp as ready to start
            __threadfence();
        }
        // printf("flag[%d] ready\n",warp + 1);
    }
    __syncthreads(); // Ensure flag updates are visible

    // 5.3 For non-zero warps, compute exclusive prefix sum
    if (warp > 0)
    {
        if (!lane) // First lane of each warp
        {
            int lookback = warp;  // Walk backwards over previous warps
            int loc_excl_sum = 0; // Local exclusive prefix sum

            while (lookback > 0) // Accumulate prefix sum for all previous warps
            {
                int status;
                do
                {
                    status = flag[lookback]; // Read warp status
                    __threadfence();         // Ensure we see the latest value
                } while (status == 0);

                if (status == 2)
                {
                    loc_excl_sum += cmpOffset[lookback]; // Accumulate previous cmpOffset
                    __threadfence();
                    break;
                }
                if (status == 1)
                    loc_excl_sum += locOffset[lookback]; // Accumulate previous locOffset
                lookback--;
                __threadfence();
                // printf(" turn flag[%d]:%d\n",lookback,status);
            }
            excl_sum = loc_excl_sum; // Store exclusive prefix sum

            cmpOffset[warp] = excl_sum; // Update cmpOffset for current warp
            __threadfence();            // Ensure write is visible

            // printf("flag[%d] over1\n",warp);
            if (warp == gridDim.x - 1)
            {
                cmpOffset[warp + 1] = cmpOffset[warp] + locOffset[warp + 1]; // Update final warp cmpOffset
                __threadfence();
            }
            flag[warp] = 2; // Mark current warp as complete
            __threadfence();
        }
    }
    else
    {
        // warp == 0: explicitly set exclusive sum to zero (single writer, block-visible)
        if (!lane)
        {
            excl_sum = 0;
        }
    }
    __syncthreads(); // Synchronize to ensure cmpOffset is fully updated
    if (numDatas <= 0)
    {
        if (cmpOffset[warp + 1] <= 0)
        {
            cmpOffset[warp + 1] = cmpOffset[warp] + locOffset[warp + 1];
        }
        return;
    }
    // 5.4 Compute write position
    int outputIdxBit = excl_sum + thread_ofs - bitSize; // Global bit offset of the compressed data to be written
    int outputIdx = (outputIdxBit + 7) / 8;

    // 6. Begin writing out the compressed block

    unsigned int firstValueBits = 0;
    memcpy(&firstValueBits, &firstValue, sizeof(int));
    // if (outputIdx % 8 != 0) {
    // 6.1 Write bitSize (4 bytes)
    for (int i = 0; i < 4; i++)
    {
        output[outputIdx + i] = (bitSize >> (i * 8)) & 0xFF;
    }

    // 6.2. Write firstValue (4 bytes)
    for (int i = 0; i < 4; i++)
    {
        output[outputIdx + 4 + i] = (firstValueBits >> (i * 8)) & 0xFF;
    }

    // }
    // 6.3. Write maxDecimalPlaces and bitCount (1 byte each)
    output[outputIdx + 8] = static_cast<unsigned char>(maxDecimalPlaces);
    output[outputIdx + 9] = static_cast<unsigned char>(maxBeta);
    output[outputIdx + 10] = static_cast<unsigned char>(bitCount);

    // 6.4 Write flag1 (4 bytes, sparse-column mask)
    for (int i = 0; i < 4; i++)
    {
        output[outputIdx + 11 + i] = (flag1 >> (i * 8)) & 0xFF;
    }
    //     if(idx==0){
    //     printf("COM: bitSize:%d, firsta: %d, maxDecimalPlaces: %d, maxBeta: %d, flag1: %d, bitcount: %d \n", bitSize, firstValueBits,maxDecimalPlaces,maxBeta,flag1 ,bitCount);
    // }
    // printf("In %d  flag1 is : %llx\n",idx,flag1);
    // 6.5 Write each column
    int flag2Byte = (numByte + 7) / 8;
    int ofs = outputIdx + 15;
    // int res=0;              // remaining bits in the last byte
    for (int i = 0; i < bitCount; i++)
    {
        if ((flag1 & (1ULL << i)) != 0) // Non-zero flag bit i means this column is sparse
        {
            // 6.5.1 flag2+data
            for (int j = 0; j < flag2Byte; j++)
            {
                output[ofs++] = static_cast<unsigned char>(flag2[i][j]);
                // printf("flag2[%d][%d]:0x%llx\n",i,j,flag2[i][j]);
            }
            for (int j = 0; j < numByte; j++)
            {
                if (result[i][j])
                {
                    output[ofs++] = static_cast<unsigned char>(result[i][j]);
                }
            }
        }
        else
        {
            // 6.5.2 For dense columns, write data directly

            for (int j = 0; j < numByte; j++)
            {
                output[ofs++] = static_cast<unsigned char>(result[i][j]);
            }
        }
    }
}
// cmpSize is returned in bytes
void FalconCompressor::Falcon_compress(float *d_oriData, unsigned char *d_cmpBytes, size_t nbEle, size_t *cmpSize, cudaStream_t stream)
{

    int bsize = cmp_tblock_size;
    int gsize = (nbEle + bsize * DATA_PER_THREAD - 1) / (bsize * DATA_PER_THREAD);
    // size_t numthread = (nbEle + cmp_chunk - 1) / (cmp_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    unsigned int *d_cmpOffset;
    unsigned int *d_locOffset;
    int *d_flag;
    unsigned int glob_sync;
    cudaMallocAsync((void **)&d_cmpOffset, sizeof(unsigned int) * cmpOffSize, stream);
    cudaMemsetAsync(d_cmpOffset, 0, sizeof(unsigned int) * cmpOffSize, stream);
    cudaMallocAsync((void **)&d_locOffset, sizeof(unsigned int) * cmpOffSize, stream);
    cudaMemsetAsync(d_locOffset, 0, sizeof(unsigned int) * cmpOffSize, stream);
    cudaMallocAsync((void **)&d_flag, sizeof(int) * cmpOffSize, stream);
    cudaMemsetAsync(d_flag, 0, sizeof(int) * cmpOffSize, stream);

    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);

    // printf("run\n");
    Falcon_compress_kernel<<<gridSize, blockSize, sizeof(unsigned int) * 2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, nbEle);

    // Obtain compression ratio and move data back to CPU.
    cudaMemcpyAsync(&glob_sync, d_cmpOffset + cmpOffSize - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    *cmpSize = ((size_t)glob_sync + 7) / 8; //+ (nbEle+cmp_tblock_size*cmp_chunk-1)/(cmp_tblock_size*cmp_chunk)*(cmp_tblock_size*cmp_chunk)/32;

    cudaFreeAsync(d_cmpOffset, stream);
    cudaFreeAsync(d_locOffset, stream);
    cudaFreeAsync(d_flag, stream);
}

//bits
void FalconCompressor::Falcon_compress_stream(float *d_oriData, unsigned char *d_cmpBytes, unsigned int *d2h_async_totalBits_ptr, size_t nbEle, cudaStream_t stream)
{
    // Data blocking.
    int bsize = cmp_tblock_size;
    int gsize = (nbEle + bsize * DATA_PER_THREAD - 1) / (bsize * DATA_PER_THREAD);
    // size_t numthread = (nbEle + cmp_chunk - 1) / (cmp_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    unsigned int *d_cmpOffset;
    unsigned int *d_locOffset;
    int *d_flag;
    cudaMallocAsync((void **)&d_cmpOffset, sizeof(unsigned int) * cmpOffSize, stream);
    cudaMemsetAsync(d_cmpOffset, 0, sizeof(unsigned int) * cmpOffSize, stream);
    cudaMallocAsync((void **)&d_locOffset, sizeof(unsigned int) * cmpOffSize, stream);
    cudaMemsetAsync(d_locOffset, 0, sizeof(unsigned int) * cmpOffSize, stream);
    cudaMallocAsync((void **)&d_flag, sizeof(int) * cmpOffSize, stream);
    cudaMemsetAsync(d_flag, 0, sizeof(int) * cmpOffSize, stream);

    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);

    Falcon_compress_kernel<<<gridSize, blockSize, sizeof(unsigned int) * 2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, nbEle);

    // printf("beta %.d ",d_cmpBytes[16]);
    // Obtain compression ratio and move data back to CPU.
    cudaMemcpyAsync(d2h_async_totalBits_ptr, (d_cmpOffset + cmpOffSize - 1), sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaFreeAsync(d_cmpOffset, stream);
    cudaFreeAsync(d_locOffset, stream);
    cudaFreeAsync(d_flag, stream);
}



__global__ void compressBlockKernel(
    const float *input,
    int totalSize,
    unsigned char *output,
    uint64_t *bitSizes,
    volatile unsigned int *const __restrict__ cmpOffset, //translated comment
    volatile unsigned int *const __restrict__ locOffset, //translated comment
    volatile int *const __restrict__ flag                //， warp （
)
{
    //translated comment
    __shared__ unsigned int excl_sum; //translated comment

    //translated comment
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f; // Lane index within the warp (0-31)
    const int warp = idx >> 5;   // Warp index

    // Each thread processes DATA_PER_THREAD elements
    int startIdx = idx * DATA_PER_THREAD;
    int endIdx = min(startIdx + DATA_PER_THREAD, totalSize);
    int numDatas = endIdx - startIdx;
    uint32_t deltas[DATA_PER_THREAD];
    // int numDeltas = numDatas - 1;
    if (numDatas <= 0)
    {
        return;
    }
    //translated comment
    int maxDecimalPlaces = 0;
    int maxBeta = 0;
    int firstValue = 0;
    int bitCount = 0;

    int base_block_start_idx, base_block_end_idx;
    int quant_chunk_idx;
    //int block_idx; // ，

    int currQuant;
    int lorenQuant;
    int prevQuant;

    unsigned int thread_ofs = 0;
    float4 tmp_buffer;

    // 1. Sampling
    for (int i = 0; i < numDatas; i++)
    {
        float value = input[startIdx + i];
        float log10v = log10(std::abs(value));
        int sp = floor(log10v);

        float alpha = getDecimalPlaces(value, sp); // Number of decimal places
        float beta = alpha + sp + 1;
        maxBeta = device_max(maxBeta, beta);
        maxDecimalPlaces = device_max(maxDecimalPlaces, alpha);
    }

    uint32_t maxDelta = 0;
    firstValue = float2int(input[startIdx], maxDecimalPlaces, maxBeta); // Quantize the first value
    prevQuant = firstValue;                                             // Initialize previous quantized value
    for (int j = 0; j < (numDatas + 30) / 32; j++)
    {                                                   //translated comment
        base_block_start_idx = startIdx + j * 32 + 1;       //translated comment
        base_block_end_idx = base_block_start_idx + 32 + 1; //translated comment

        if (base_block_end_idx < totalSize)
        {
            int i = base_block_start_idx;
            /*
                tmp_buffer = reinterpret_cast<const float4 *>(input)[i / 4];
                quant_chunk_idx = j * 32 + (i % 32); //translated comment

                //translated comment
                if (i == startIdx)
                { //translated comment

                    deltas[quant_chunk_idx] = 0; //translated comment
                }
                else
                {
                    currQuant = float2int(tmp_buffer.x, maxDecimalPlaces, maxBeta); //translated comment
                    lorenQuant = currQuant - prevQuant;                             //translated comment

                    deltas[quant_chunk_idx] = zigzag_encode_cuda(lorenQuant);
                    prevQuant = currQuant;                                           //translated comment
                    maxDelta = device_max_uint32(maxDelta, deltas[quant_chunk_idx]); //translated comment
                }

                //translated comment
                currQuant = float2int(tmp_buffer.y, maxDecimalPlaces, maxBeta);
                lorenQuant = currQuant - prevQuant;

                deltas[quant_chunk_idx + 1] = zigzag_encode_cuda(lorenQuant);
                prevQuant = currQuant;
                maxDelta = device_max_uint32(maxDelta, deltas[quant_chunk_idx + 1]);

                //translated comment
                currQuant = float2int(tmp_buffer.z, maxDecimalPlaces, maxBeta);
                lorenQuant = currQuant - prevQuant;

                deltas[quant_chunk_idx + 2] = zigzag_encode_cuda(lorenQuant);
                prevQuant = currQuant;
                maxDelta = device_max_uint32(maxDelta, deltas[quant_chunk_idx + 2]);

                //translated comment
                currQuant = float2int(tmp_buffer.w, maxDecimalPlaces, maxBeta);
                lorenQuant = currQuant - prevQuant;

                deltas[quant_chunk_idx + 3] = zigzag_encode_cuda(lorenQuant);
                prevQuant = currQuant;
                maxDelta = device_max_uint32(maxDelta, deltas[quant_chunk_idx + 3]);
                i += 4;
            */

            #pragma unroll 8 //translated comment
            for (; i < base_block_end_idx; i += 4)
            {

                // tmp_buffer = reinterpret_cast<const float4 *>(input)[i / 4];
                //quant_chunk_idx = j * 32 + (i % 32); //

                tmp_buffer = reinterpret_cast<const float4*>(input+1)[(i-1) / 4];
                quant_chunk_idx = j * 32 + ((i-1) % 32); //translated comment

                currQuant = float2int(tmp_buffer.x, maxDecimalPlaces, maxBeta); //translated comment
                lorenQuant = currQuant - prevQuant;                             //translated comment

                deltas[quant_chunk_idx] = zigzag_encode_cuda(lorenQuant);
                prevQuant = currQuant;                                           //translated comment
                maxDelta = device_max_uint32(maxDelta, deltas[quant_chunk_idx]); //translated comment
                // }

                //translated comment
                currQuant = float2int(tmp_buffer.y, maxDecimalPlaces, maxBeta);
                lorenQuant = currQuant - prevQuant;

                deltas[quant_chunk_idx + 1] = zigzag_encode_cuda(lorenQuant);
                prevQuant = currQuant;
                maxDelta = device_max_uint32(maxDelta, deltas[quant_chunk_idx + 1]);

                //translated comment
                currQuant = float2int(tmp_buffer.z, maxDecimalPlaces, maxBeta);
                lorenQuant = currQuant - prevQuant;

                deltas[quant_chunk_idx + 2] = zigzag_encode_cuda(lorenQuant);
                prevQuant = currQuant;
                maxDelta = device_max_uint32(maxDelta, deltas[quant_chunk_idx + 2]);

                //translated comment
                currQuant = float2int(tmp_buffer.w, maxDecimalPlaces, maxBeta);
                lorenQuant = currQuant - prevQuant;

                deltas[quant_chunk_idx + 3] = zigzag_encode_cuda(lorenQuant);
                prevQuant = currQuant;
                maxDelta = device_max_uint32(maxDelta, deltas[quant_chunk_idx + 3]);
            }
        }
        else
        {
            //translated comment
            if (base_block_start_idx >= endIdx)
            {
                //， absQuant 0
                quant_chunk_idx = j * 32 + (base_block_start_idx % 32);
                for (int i = quant_chunk_idx; i < quant_chunk_idx + 32; i++)
                    deltas[i] = 0;
            }
            else
            {
                //translated comment
                int remainbEle = totalSize - base_block_start_idx; //translated comment
                int zeronbEle = base_block_end_idx - totalSize;    //translated comment

                //translated comment
                for (int i = base_block_start_idx; i < base_block_start_idx + remainbEle; i++)
                {
                    if (i == startIdx)
                    {
                        deltas[0] = 0;
                        continue;
                    }
                    quant_chunk_idx = j * 32 + (i % 32);
                    currQuant = float2int(input[i], maxDecimalPlaces, maxBeta);

                    lorenQuant = currQuant - prevQuant;

                    deltas[quant_chunk_idx] = zigzag_encode_cuda(lorenQuant);
                    prevQuant = currQuant;
                    maxDelta = device_max_uint32(maxDelta, deltas[quant_chunk_idx]);
                }

                quant_chunk_idx = j * 32 + (totalSize % 32);
                for (int i = quant_chunk_idx; i < quant_chunk_idx + zeronbEle; i++)
                    deltas[i] = 0;
            }
        }
    }

    bitCount = maxDelta > 0 ? 32 - __clz(maxDelta) : 1; //translated comment
    bitCount = min(bitCount, (int)MAX_BITCOUNT);

    int numByte = (numDatas-1 + 7) / 8;
    // uint8_t result[64][128];
    uint8_t result[32][128] = {}; // Zero-initialize 2D array

    // Traverse each bit-plane
    for (int i = 0; i < bitCount; ++i)
    { //translated comment
        int j = 0;
        while (j + 8 < numDatas-1) //translated comment
        {
            int byteIndex = j / 8; //bit
            result[i][byteIndex] = result[i][byteIndex] |
                                   (((deltas[j] >> (bitCount - 1 - i)) & 1) << (7)) |
                                   (((deltas[j + 1] >> (bitCount - 1 - i)) & 1) << (6)) |
                                   (((deltas[j + 2] >> (bitCount - 1 - i)) & 1) << (5)) |
                                   (((deltas[j + 3] >> (bitCount - 1 - i)) & 1) << (4)) |
                                   (((deltas[j + 4] >> (bitCount - 1 - i)) & 1) << (3)) |
                                   (((deltas[j + 5] >> (bitCount - 1 - i)) & 1) << (2)) |
                                   (((deltas[j + 6] >> (bitCount - 1 - i)) & 1) << (1)) |
                                   (((deltas[j + 7] >> (bitCount - 1 - i)) & 1) << (0));
            j += 8;
        }
        for (; j < numDatas-1; ++j)
        {                          //numBytes
            int byteIndex = j / 8; //bit
            int bitIndex = j % 8;  //bit

            //bit
            result[i][byteIndex] |= (((deltas[j] >> (bitCount - 1 - i)) & 1) << (7 - bitIndex));
        }
    }

    //4.2 ， ， bitsize
    uint64_t bitSize = 32 +   // bitsize
                       32 +   // firstValue
                       8ULL + // maxDecimalPlaces
                       8ULL + // maxBeta
                       8ULL + // bitCount
                       32;    // flag1

    uint32_t flag1 = 0;    //translated comment
    uint8_t flag2[32][16]; //, 1024 ， 1024bit， 128byte,
    memset(flag2, 0, sizeof(flag2));
    int BITS_PER_THREAD = 4;
    for (int i = 0; i < bitCount; i += BITS_PER_THREAD)
    { //translated comment
        for (int b = 0; b < BITS_PER_THREAD && (i + b) < bitCount; ++b)
        {
            int bit = i + b;
            int b0 = 0;
            int b1 = 0;
            for (int j = 0; j < numByte; j++)
            {
                int m_byte = j / 8;
                int m_bit = j % 8;
                uint8_t current_result = result[bit][j];
                b0 += (current_result == 0);
                b1 += (current_result != 0);
                flag2[bit][m_byte] |= (current_result != 0) << m_bit;    //translated comment
                flag2[bit][m_byte] &= ~((current_result == 0) << m_bit); //translated comment
            }
            //translated comment
            uint32_t is_sparse = ((numByte + 7) / 8 + b1) < numByte;
            flag1 |= (is_sparse << bit);
            flag1 &= ~((!is_sparse) << bit);
            bitSize += is_sparse ? ((numByte + 7) / 8 + b1) * 8 : 8 * numByte;
            //flag2 +b1 numByte*8
        }
    }
    //translated comment
    thread_ofs += bitSize; //bitSize bit

//5.1. Warp( ) ，
#pragma unroll 5
    for (int i = 1; i < 32; i <<= 1)
    {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if (lane >= i)
            thread_ofs += tmp; //translated comment
    }
    __syncthreads(); //translated comment

    //5.2 Warp( ) locOffset flag
    if (lane == 31)
    {
        locOffset[warp + 1] = thread_ofs; // Update local offset for the next warp
        __threadfence();                  // Ensure global write has completed
        if (warp == 0)
        {
            flag[0] = 2; // Mark first warp as having completed prefix sum
            __threadfence();
            flag[1] = 1; // Mark next warp as ready to start
            __threadfence();
        }
        else
        {
            flag[warp + 1] = 1; // Mark next warp as ready to start
            __threadfence();
        }
    }
    __syncthreads(); //， flag

    //5.3 warp， （ ）
    if (warp > 0)
    {
        if (!lane) // First lane of each warp
        {
            int lookback = warp;  // Walk backwards over previous warps
            int loc_excl_sum = 0; // Local exclusive prefix sum

            while (lookback > 0) // Accumulate prefix sum for all previous warps
            {
                int status;
                do
                {
                    status = flag[lookback]; //warp
                    __threadfence();         //translated comment
                } while (status == 0);

                if (status == 2)
                {
                    loc_excl_sum += cmpOffset[lookback]; //warp cmpOffset
                    __threadfence();
                    break;
                }
                if (status == 1)
                    loc_excl_sum += locOffset[lookback]; //warp locOffset
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum; //translated comment
            //2.3 cmpOffset
            cmpOffset[warp] = excl_sum; //warp cmpOffset
            __threadfence();            //translated comment

            if (warp == gridDim.x - 1)
            {
                cmpOffset[warp + 1] = cmpOffset[warp] + locOffset[warp + 1]; //warp cmpOffset
                __threadfence();
            }
            flag[warp] = 2; //warp
            __threadfence();
        }
    }
    else
    {
        //warp==0： 0（ ，block ）
        if (!lane)
        {
            excl_sum = 0;
        }
    }
    __syncthreads(); // Synchronize to ensure cmpOffset is fully updated

    // 5.4 Compute write position
    int outputIdxBit = excl_sum + thread_ofs - bitSize; // Global bit offset of the compressed data to be written
    int outputIdx = (outputIdxBit + 7) / 8;

    bitSizes[idx] = bitSize;

    unsigned int firstValueBits = 0;
    memcpy(&firstValueBits, &firstValue, sizeof(int));
    //6.1 bitSize (4 )
    for (int i = 0; i < 4; i++)
    {
        output[outputIdx + i] = (bitSize >> (i * 8)) & 0xFF;
    }

    //6.2. firstValue (8 )
    for (int i = 0; i < 4; i++)
    {
        output[outputIdx + 4 + i] = (firstValueBits >> (i * 8)) & 0xFF;
    }
    // 6.3. Write maxDecimalPlaces and maxBeta (1 byte each)
    output[outputIdx + 8] = static_cast<unsigned char>(maxDecimalPlaces);
    output[outputIdx + 9] = static_cast<unsigned char>(maxBeta);
    output[outputIdx + 10] = static_cast<unsigned char>(bitCount);
    //     if(bid==0){
    //     printf("COM: bitSize:%d, first: %d, maxDecimalPlaces: %d, maxBeta: %d, bitcount: %d \n",bitSize, firstValue,maxDecimalPlaces,maxBeta,bitCount);
    // }

    //6.4 flag1(4 )
    for (int i = 0; i < 4; i++)
    {
        output[outputIdx + 11 + i] = (flag1 >> (i * 8)) & 0xFF;
    }
    //translated comment
    int flag2Byte = (numByte + 7) / 8;
    int ofs = outputIdx + 15;
    for (int i = 0; i < bitCount; i++)
    {
        if ((flag1 & (1ULL << i)) != 0) // Non-zero flag bit i means this column is sparse
        {
            //6.5.1 flag2+data
            for (int j = 0; j < flag2Byte; j++)
            {
                output[ofs++] = static_cast<unsigned char>(flag2[i][j]);
            }
            for (int j = 0; j < numByte; j++)
            {
                if (result[i][j])
                {
                    output[ofs++] = static_cast<unsigned char>(result[i][j]);
                }
            }
        }
        else
        {
            // 6.5.2 For dense columns, write data directly

            for (int j = 0; j < numByte; j++)
            {
                output[ofs++] = static_cast<unsigned char>(result[i][j]);
            }
        }
    }
}

// Initialize device memory
void FalconCompressor::setupDeviceMemory(
    const std::vector<float> &input,
    float *&d_input,
    unsigned char *&d_output,
    uint64_t *&d_bitSizes)
{
    size_t inputSize = input.size();
    int numBlocks = (inputSize + DATA_PER_THREAD - 1) / (DATA_PER_THREAD);

    // Allocate device memory for input
    cudaCheckError(cudaMalloc((void **)&d_input, inputSize * sizeof(float)));
    cudaCheckError(cudaMemcpy(d_input, input.data(), inputSize * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate device memory for output
    cudaCheckError(cudaMalloc((void **)&d_output, numBlocks * MAX_BYTES_PER_BLOCK * sizeof(unsigned char)));

    // Allocate device memory for bitSizes
    cudaCheckError(cudaMalloc((void **)&d_bitSizes, numBlocks * sizeof(uint64_t)));
}

// Function to free device memory
void FalconCompressor::freeDeviceMemory(
    float *d_input,
    unsigned char *d_output,
    uint64_t *d_bitSizes)
{
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_bitSizes);
}

// Main compression function
void FalconCompressor::compress(const std::vector<float> &input, std::vector<unsigned char> &output)
{
    // std::cout<<"begin1\n";
    size_t inputSize = input.size();
    if (inputSize == 0)
        return;

    int blockSize = BLOCK_SIZE_G;                                                                     // 32 threads per block
    size_t numBlocks = (inputSize + blockSize * DATA_PER_THREAD - 1) / (blockSize * DATA_PER_THREAD); // Number of thread blocks
    size_t numthread = (inputSize + DATA_PER_THREAD - 1) / (DATA_PER_THREAD);                         // Number of data blocks (threads)
    float *d_input = nullptr;
    unsigned char *d_output = nullptr;
    uint64_t *d_bitSizes = nullptr;

    unsigned int *d_cmpOffset;
    unsigned int *d_locOffset;
    int *d_flag;
    int cmpOffSize = numBlocks + 1;
    cudaMalloc((void **)&d_cmpOffset, sizeof(unsigned int) * cmpOffSize);
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int) * cmpOffSize);

    cudaMalloc((void **)&d_locOffset, sizeof(unsigned int) * cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(unsigned int) * cmpOffSize);

    cudaMalloc((void **)&d_flag, sizeof(int) * cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int) * cmpOffSize);
    // Allocate device memory
    setupDeviceMemory(input, d_input, d_output, d_bitSizes);

    size_t sharedMemSize = 64; // Ensure SharedMemory is defined correctly
    // std::cout<<"begin2\n";

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaCheckError(cudaEventCreate(&start));
    cudaCheckError(cudaEventCreate(&stop));

    // Record start event
    cudaCheckError(cudaEventRecord(start));

    // Launch kernel
    compressBlockKernel<<<numBlocks, blockSize, sharedMemSize>>>(
        d_input,
        inputSize,
        d_output,
        d_bitSizes,
        d_cmpOffset,
        d_locOffset,
        d_flag);
    // Check for errors
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());
    // std::cout<<"end2\n";

    //translated comment
    cudaCheckError(cudaEventRecord(stop));
    cudaCheckError(cudaEventSynchronize(stop)); //translated comment

    //translated comment
    float milliseconds = 0;
    cudaCheckError(cudaEventElapsedTime(&milliseconds, start, stop));

    //translated comment
    size_t dataSizeBytes = input.size() * sizeof(float); //translated comment
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
    // std::cout<<"end3\n";

    //translated comment
    std::vector<uint64_t> offsets(numthread, 0);
    uint64_t totalCompressedBits = 0;
    for (size_t i = 0; i < numthread; i++)
    {
        offsets[i] = totalCompressedBits;
        totalCompressedBits += bitSizes[i];
    }

    uint64_t totalCompressedBytes = (totalCompressedBits + 7) / 8; //translated comment

    //translated comment
    output.resize(totalCompressedBytes, 0);
    //d_output
    std::vector<unsigned char> tempOutput(totalCompressedBytes);
    cudaCheckError(cudaMemcpy(tempOutput.data(), d_output, totalCompressedBytes * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    output = std::move(tempOutput);

    freeDeviceMemory(d_input, d_output, d_bitSizes);
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}
