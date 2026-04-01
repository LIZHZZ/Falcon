// Falcon_pipeline.cu
// Pipeline compression and decompression implementation
// Path: Falcon/src/gpu/Falcon_pipeline.cu

#include "Falcon_pipeline.cuh"
#include <algorithm>
#include <iostream>
#include <cstring>

// Constructor
FalconPipeline::FalconPipeline(int numStreams) : NUM_STREAMS(numStreams) {
}

// Destructor
FalconPipeline::~FalconPipeline() {
}

// Helper: create a CompressedData structure from a CompressionResult
CompressedData FalconPipeline::createCompressedData(
    const CompressionResult& compResult,
    const ProcessedData& originalData) {
    
    CompressedData compData;
    compData.cmpBytes = originalData.cmpBytes;
    compData.totalCompressedSize = compResult.analysis.total_compressed_size;
    compData.totalElements = originalData.nbEle;
    compData.chunkSizes = compResult.chunkSizes;
    compData.chunkElementCounts = compResult.chunkElementCounts;
    compData.totalChunks = compResult.totalChunks;
    
    return compData;
}

// Cleanup helper for host-side resources
void cleanup_data(ProcessedData &data) {
    if (data.oriData != nullptr) {
        cudaFreeHost(data.oriData);
        data.oriData = nullptr;
    }

    if (data.cmpBytes != nullptr) {
        cudaFreeHost(data.cmpBytes);
        data.cmpBytes = nullptr;
    }

        if (data.decData != nullptr) {
        cudaFreeHost(data.decData);
        data.decData = nullptr;
    }
}

// Standard implementation

// Run compression pipeline using the member NUM_STREAMS
CompressionResult FalconPipeline::executeCompressionPipeline(
    ProcessedData &data,
    size_t chunkSize) {
    
    return executeCompressionPipelineImpl(data, chunkSize, NUM_STREAMS);
}

// Run compression pipeline with an explicit number of streams
CompressionResult FalconPipeline::executeCompressionPipeline(
    ProcessedData &data,
    size_t chunkSize,
    int numStreams) {
    
    return executeCompressionPipelineImpl(data, chunkSize, numStreams);
}

// Internal implementation of the compression pipeline
CompressionResult FalconPipeline::executeCompressionPipelineImpl(
    ProcessedData &data,
    size_t chunkSize,
    int numStreams) {
    
    cudaDeviceSynchronize();
    
    // Create events to record the overall timeline
    cudaEvent_t global_start_event, global_end_event;
    cudaEventCreate(&global_start_event);
    cudaEventCreate(&global_end_event);
    
    cudaEvent_t init_start_event, init_end_event;
    cudaEventCreate(&init_start_event);
    cudaEventCreate(&init_end_event);
    
    // Initialization
    cudaDeviceSynchronize();
    cudaEventRecord(init_start_event);
    
    // Compute total number of chunks
    size_t totalChunks = (data.nbEle + chunkSize - 1) / chunkSize;
    if(totalChunks == 1) {
        chunkSize = data.nbEle;
    }
    
    printf("totalChunks: %zu\n", totalChunks);
    
    // Host-side memory allocation
    unsigned int *locCmpSize;
    cudaCheckError(cudaHostAlloc((void**)&locCmpSize, 
        sizeof(unsigned int) * (totalChunks + 2), cudaHostAllocDefault));
    
    // Set guard values
    locCmpSize[0] = 0xDEADBEEF;
    locCmpSize[totalChunks + 1] = 0xCAFEBABE;
    
    unsigned int *h_cmp_offset;
    cudaCheckError(cudaHostAlloc((void**)&h_cmp_offset, 
        sizeof(unsigned int) * (totalChunks + 1), cudaHostAllocDefault));
    h_cmp_offset[0] = 0;
    
    bool *of_rd = new bool[totalChunks + numStreams]();
    of_rd[0] = true;
    
    size_t *chunkElementCounts;
    cudaCheckError(cudaHostAlloc((void**)&chunkElementCounts, 
        sizeof(size_t) * totalChunks, cudaHostAllocDefault));
    
    std::vector<size_t> chunkSizes(totalChunks);
    std::vector<size_t> chunkElementCountsVec(totalChunks);
    
    // Create stream pool and events
    const int MAX_EVENTS_PER_TYPE = totalChunks + numStreams;
    
    cudaStream_t *streams = new cudaStream_t[numStreams];
    cudaEvent_t *evSize = new cudaEvent_t[MAX_EVENTS_PER_TYPE];
    cudaEvent_t *evData = new cudaEvent_t[MAX_EVENTS_PER_TYPE];
    Stage *stage = new Stage[numStreams];
    
    for (int i = 0; i < numStreams; ++i) {
        cudaCheckError(cudaStreamCreate(&streams[i]));
        stage[i] = IDLE;
    }
    
    for (int i = 0; i < MAX_EVENTS_PER_TYPE; ++i) {
        cudaCheckError(cudaEventCreateWithFlags(&evSize[i], cudaEventDisableTiming));
        cudaCheckError(cudaEventCreateWithFlags(&evData[i], cudaEventDisableTiming));
    }
    
    // Allocate fixed device buffers for each stream
    double **d_in = new double*[numStreams];
    unsigned char **d_out = new unsigned char*[numStreams];
    
    for (int i = 0; i < numStreams; ++i) {
        cudaCheckError(cudaMalloc(&d_in[i], chunkSize * sizeof(double)));
        cudaCheckError(cudaMalloc(&d_out[i], chunkSize * sizeof(double)));
    }
    
    cudaEventRecord(init_end_event);
    cudaEventSynchronize(init_end_event);
    
    // Main loop: poll streams until all data is processed
    size_t processedEle = 0;
    int active = 0;
    size_t totalCmpSize = 0;
    size_t completedChunks = 0;
    
    cudaDeviceSynchronize();
    cudaEventRecord(global_start_event);
    
    std::vector<int> chunkIDX(numStreams);
    
    while (processedEle < data.nbEle || active > 0) {
        int progress = 0;
        
        for (int s = 0; s < numStreams; ++s) {
            switch (stage[s]) {
                case IDLE:
                    if (processedEle < data.nbEle) {
                        size_t todo = std::min(chunkSize, data.nbEle - processedEle);
                        if(todo == 0) continue;
                        
                        progress = 1;
                        chunkIDX[s] = completedChunks;
                        completedChunks++;
                        
                        chunkElementCounts[chunkIDX[s]] = todo;
                        
                        // Asynchronous host-to-device copy
                        cudaCheckError(cudaMemcpyAsync(
                            d_in[s],
                            data.oriData + processedEle,
                            todo * sizeof(double),
                            cudaMemcpyHostToDevice,
                            streams[s]));
                        
                        // Launch compression kernel
                        FalconCompressor::Falcon_compress_stream(
                            d_in[s],
                            d_out[s],
                            &locCmpSize[chunkIDX[s] + 1],
                            todo,
                            streams[s]);
                        
                        active += 1;
                        processedEle += todo;
                        stage[s] = SIZE_PENDING;
                        cudaCheckError(cudaEventRecord(evSize[chunkIDX[s]], streams[s]));
                    }
                    break;
                    
                case SIZE_PENDING:
                    if(cudaEventQuery(evSize[chunkIDX[s]]) == cudaSuccess && 
                       of_rd[chunkIDX[s]]) {
                        
                        if (locCmpSize[0] != 0xDEADBEEF || 
                            locCmpSize[totalChunks + 1] != 0xCAFEBABE) {
                            printf("Error: guard values of locCmpSize were overwritten!\n");
                        }
                        
                        progress = 1;
                        int idx = chunkIDX[s];
                        
                        unsigned int compressedBits = locCmpSize[idx + 1];
                        unsigned int compressedBytes = (compressedBits + 7) / 8;
                        
                        // Asynchronous device-to-host copy of compressed results
                        cudaCheckError(cudaMemcpyAsync(
                            data.cmpBytes + h_cmp_offset[idx],
                            d_out[s],
                            compressedBytes,
                            cudaMemcpyDeviceToHost,
                            streams[s]));
                        
                        h_cmp_offset[idx + 1] = h_cmp_offset[idx] + compressedBytes;
                        of_rd[idx + 1] = true;
                        chunkSizes[idx] = compressedBytes;
                        chunkElementCountsVec[idx] = chunkElementCounts[idx];
                        
                        cudaCheckError(cudaEventRecord(evData[chunkIDX[s]], streams[s]));
                        stage[s] = DATA_PENDING;
                    }
                    break;
                    
                case DATA_PENDING:
                    if (cudaEventQuery(evData[chunkIDX[s]]) == cudaSuccess) {
                        unsigned int compressedBytes = (locCmpSize[chunkIDX[s] + 1] + 7) / 8;
                        totalCmpSize += compressedBytes;
                        progress = 1;
                        stage[s] = IDLE;
                        active -= 1;
                    }
                    break;
            }
        }
        
        // Avoid busy waiting
        if (!progress) {
            for (int i = 0; i < numStreams; i++) {
                cudaCheckError(cudaStreamSynchronize(streams[i]));
            }
        }
    }
    
    // Wait for all streams to finish
    for (int i = 0; i < std::min(numStreams, (int)totalChunks); i++) {
        cudaCheckError(cudaStreamSynchronize(streams[i]));
        cudaCheckError(cudaEventSynchronize(evData[i]));
    }
    
    cudaEventRecord(global_end_event);
    cudaEventSynchronize(global_end_event);
    
    // Compute total time
    float totalTime;
    cudaEventElapsedTime(&totalTime, global_start_event, global_end_event);
    
    // Compute compression ratio
    double compressionRatio = totalCmpSize / static_cast<double>(data.nbEle * sizeof(double));
    
    // Create analysis result
    PipelineAnalysis analysis;
    analysis.compression_ratio = compressionRatio;
    analysis.total_compressed_size = totalCmpSize;
    analysis.total_size = data.nbEle * sizeof(double) / 1024.0 / 1024.0;
    analysis.comp_time = totalTime;
    analysis.comp_throughout = (data.nbEle * sizeof(double) / 1024.0 / 1024.0 / 1024.0) / 
                              (totalTime / 1000.0);
    analysis.chunk_size = chunkSize;
    *data.cmpSize = totalCmpSize;
    
    // Free device memory
    for (int i = 0; i < numStreams; i++) {
        cudaCheckError(cudaFree(d_out[i]));
        cudaCheckError(cudaFree(d_in[i]));
    }
    
    // Destroy streams and events
    for (int i = 0; i < numStreams; i++) {
        cudaCheckError(cudaStreamDestroy(streams[i]));
    }
    for (int i = 0; i < MAX_EVENTS_PER_TYPE; i++) {
        cudaCheckError(cudaEventDestroy(evSize[i]));
        cudaCheckError(cudaEventDestroy(evData[i]));
    }
    
    // Free host memory
    cudaCheckError(cudaFreeHost(locCmpSize));
    cudaCheckError(cudaFreeHost(h_cmp_offset));
    cudaCheckError(cudaFreeHost(chunkElementCounts));
    delete[] of_rd;
    
    // Destroy global events
    cudaCheckError(cudaEventDestroy(global_start_event));
    cudaCheckError(cudaEventDestroy(global_end_event));
    cudaCheckError(cudaEventDestroy(init_start_event));
    cudaCheckError(cudaEventDestroy(init_end_event));
    
    // Free dynamically allocated arrays
    delete[] streams;
    delete[] evSize;
    delete[] evData;
    delete[] stage;
    delete[] d_in;
    delete[] d_out;
    
    // Create and return the final result
    CompressionResult result;
    result.analysis = analysis;
    result.chunkSizes = std::move(chunkSizes);
    result.chunkElementCounts = std::move(chunkElementCountsVec);
    result.totalChunks = completedChunks;
    
    return result;
}

// Run decompression pipeline using the member NUM_STREAMS
PipelineAnalysis FalconPipeline::executeDecompressionPipeline(
    const CompressionResult& compResult,
    ProcessedData &decompData,
    bool visualize) {
    
    return executeDecompressionPipelineImpl(compResult, decompData, NUM_STREAMS, visualize);
}

// Run decompression pipeline with an explicit number of streams
PipelineAnalysis FalconPipeline::executeDecompressionPipeline(
    const CompressionResult& compResult,
    ProcessedData &decompData,
    int numStreams,
    bool visualize) {
    
    return executeDecompressionPipelineImpl(compResult, decompData, numStreams, visualize);
}

// Internal implementation of the decompression pipeline
PipelineAnalysis FalconPipeline::executeDecompressionPipelineImpl(
    const CompressionResult& compResult,
    ProcessedData &decompData,
    int numStreams,
    bool visualize) {
    
    CompressedData compData = createCompressedData(compResult, decompData);
    
    cudaDeviceSynchronize();
    
    cudaEvent_t global_start_event, global_end_event;
    cudaEventCreate(&global_start_event);
    cudaEventCreate(&global_end_event);
    
    // Host-side memory allocation
    size_t *streamChunkIds;
    cudaCheckError(cudaHostAlloc((void**)&streamChunkIds, 
        sizeof(size_t) * numStreams, cudaHostAllocDefault));
    
    size_t *streamOutputOffsets;
    cudaCheckError(cudaHostAlloc((void**)&streamOutputOffsets, 
        sizeof(size_t) * numStreams, cudaHostAllocDefault));
    
    // Create stream pool and events
    cudaStream_t *streams = new cudaStream_t[numStreams];
    cudaEvent_t *evSize = new cudaEvent_t[numStreams];
    cudaEvent_t *evData = new cudaEvent_t[numStreams];
    Stage *stage = new Stage[numStreams];
    
    for (int i = 0; i < numStreams; ++i) {
        cudaCheckError(cudaStreamCreate(&streams[i]));
        cudaCheckError(cudaEventCreateWithFlags(&evSize[i], cudaEventDisableTiming));
        cudaCheckError(cudaEventCreateWithFlags(&evData[i], cudaEventDisableTiming));
        stage[i] = IDLE;
    }
    
    // Compute maximum chunk size
    size_t maxChunkSize = 0;
    size_t maxCompressedSize = 0;
    for (size_t i = 0; i < compData.totalChunks; ++i) {
        maxChunkSize = std::max(maxChunkSize, compData.chunkElementCounts[i]);
        maxCompressedSize = std::max(maxCompressedSize, compData.chunkSizes[i]);
    }
    
    // Allocate device buffers for each stream
    unsigned char **d_in = new unsigned char*[numStreams];
    double **d_out = new double*[numStreams];
    
    for (int i = 0; i < numStreams; ++i) {
        cudaCheckError(cudaMalloc(&d_in[i], maxCompressedSize));
        cudaCheckError(cudaMalloc(&d_out[i], maxChunkSize * sizeof(double)));
    }
    
    // Main loop
    size_t processedChunks = 0;
    size_t processedElements = 0;
    int active = 0;
    size_t totalDecompSize = 0;
    
    std::vector<size_t> compressedDataOffsets(compData.totalChunks);
    compressedDataOffsets[0] = 0;
    for (size_t i = 0; i < compData.totalChunks - 1; ++i) {
        compressedDataOffsets[i + 1] = compressedDataOffsets[i] + compData.chunkSizes[i];
    }
    
    cudaDeviceSynchronize();
    cudaEventRecord(global_start_event);
    
    while (processedChunks < compData.totalChunks || active > 0) {
        for (int s = 0; s < numStreams; ++s) {
            switch (stage[s]) {
                case IDLE:
                    if (processedChunks < compData.totalChunks) {
                        active += 1;
                        size_t chunkId = processedChunks;
                        size_t currentChunkElements = compData.chunkElementCounts[chunkId];
                        size_t currentChunkCompressedSize = compData.chunkSizes[chunkId];
                        
                        streamChunkIds[s] = chunkId;
                        streamOutputOffsets[s] = processedElements;
                        
                        size_t compressedDataOffset = compressedDataOffsets[chunkId];
                        
                        // Asynchronous host-to-device copy of compressed data
                        cudaCheckError(cudaMemcpyAsync(
                            d_in[s],
                            compData.cmpBytes + compressedDataOffset,
                            currentChunkCompressedSize,
                            cudaMemcpyHostToDevice,
                            streams[s]));
                        
                        // Launch decompression kernel
                        FalconDecompressor GDFD;
                        GDFD.Falcon_decompress_stream_optimized(
                            d_out[s],
                            d_in[s],
                            currentChunkElements,
                            currentChunkCompressedSize,
                            streams[s]);
                        
                        cudaCheckError(cudaEventRecord(evSize[s], streams[s]));
                        
                        processedChunks++;
                        processedElements += currentChunkElements;
                        
                        size_t outputOffset = streamOutputOffsets[s];
                        
                        // Asynchronous device-to-host copy of decompressed data
                        cudaCheckError(cudaMemcpyAsync(
                            decompData.decData + outputOffset,
                            d_out[s],
                            currentChunkElements * sizeof(double),
                            cudaMemcpyDeviceToHost,
                            streams[s]));
                        
                        cudaCheckError(cudaEventRecord(evData[s], streams[s]));
                        stage[s] = DATA_PENDING;
                    }
                    break;
                    
                case DATA_PENDING:
                    if (cudaEventQuery(evData[s]) == cudaSuccess) {
                        size_t chunkId = streamChunkIds[s];
                        size_t currentChunkElements = compData.chunkElementCounts[chunkId];
                        totalDecompSize += currentChunkElements * sizeof(double);
                        
                        stage[s] = IDLE;
                        active -= 1;
                    }
                    break;
                    
                default:
                    break;
            }
        }
    }
    
    // Wait for all streams to finish
    for (int i = 0; i < numStreams; i++) {
        cudaCheckError(cudaStreamSynchronize(streams[i]));
        cudaCheckError(cudaEventSynchronize(evData[i]));
    }
    
    cudaEventRecord(global_end_event);
    cudaEventSynchronize(global_end_event);
    
    // Compute total time
    float totalTime;
    cudaEventElapsedTime(&totalTime, global_start_event, global_end_event);
    
    // Create and return analysis result
    PipelineAnalysis result;
    result.compression_ratio = totalDecompSize / static_cast<double>(compData.totalCompressedSize);
    result.total_compressed_size = compData.totalCompressedSize / 1024.0 / 1024.0;
    result.total_size = totalDecompSize / 1024.0 / 1024.0;
    result.decomp_time = totalTime;
    result.decomp_throughout = (totalDecompSize / 1024.0 / 1024.0 / 1024.0) / (totalTime / 1000.0);
    
    // Optionally verify decompressed data
    if (visualize) {
        for(int z=0,tmp=0,k=0;z<3;z++)//translated comment
        {   
            for(size_t i=0,j=tmp;i<3;i++)//， block
            {
                while(decompData.oriData[j]==decompData.decData[j]&&j<processedElements)//translated comment
                {
                    j++;
                    if(j>=processedElements)
                    {
                        // printf("success!!\n");   
                        break;
                    }
                }
                if(j>=processedElements)
                {
                    printf("success!!\n");   
                    break;
                }
                while(tmp<=j)
                {
                    tmp+=compData.chunkElementCounts[k];
                    k++;
                }
                // printf("chunk:%d,idx: %d ,ori: %.16f , dec: %.16f \n",k-1,j,decompData.oriData[j],decompData.decData[j]);
                j++;
            }
        }
    
    }
    
    // Free device memory
    for (int i = 0; i < numStreams; i++) {
        cudaCheckError(cudaFree(d_out[i]));
        cudaCheckError(cudaFree(d_in[i]));
    }
    
    // Destroy streams and events
    for (int i = 0; i < numStreams; i++) {
        cudaCheckError(cudaEventDestroy(evSize[i]));
        cudaCheckError(cudaEventDestroy(evData[i]));
        cudaCheckError(cudaStreamDestroy(streams[i]));
    }
    
    // Free host memory
    cudaCheckError(cudaFreeHost(streamChunkIds));
    cudaCheckError(cudaFreeHost(streamOutputOffsets));
    
    // Destroy global events
    cudaCheckError(cudaEventDestroy(global_start_event));
    cudaCheckError(cudaEventDestroy(global_end_event));
    
    // Free dynamically allocated arrays
    delete[] streams;
    delete[] evSize;
    delete[] evData;
    delete[] stage;
    delete[] d_in;
    delete[] d_out;
    
    return result;
}



// NOPACK---------------------------------------------------------------------


CompressionResult FalconPipeline::executeCompressionPipelineNoPack(
    ProcessedData &data,
    size_t chunkSize) {
    
    return executeCompressionPipelineImpl_NoPack(data, chunkSize, NUM_STREAMS);
}


//translated comment
CompressionResult FalconPipeline::executeCompressionPipelineImpl_NoPack(
    ProcessedData &data,
    size_t chunkSize,
    int numStreams) {
    
    cudaDeviceSynchronize();
    
    //translated comment
    cudaEvent_t global_start_event, global_end_event;
    cudaEventCreate(&global_start_event);
    cudaEventCreate(&global_end_event);
    
    cudaEvent_t init_start_event, init_end_event;
    cudaEventCreate(&init_start_event);
    cudaEventCreate(&init_end_event);
    
    //translated comment
    cudaDeviceSynchronize();
    cudaEventRecord(init_start_event);
    
    //chunk
    size_t totalChunks = (data.nbEle + chunkSize - 1) / chunkSize;
    if(totalChunks == 1) {
        chunkSize = data.nbEle;
    }
    
    printf("totalChunks: %zu\n", totalChunks);
    
    //translated comment
    unsigned int *locCmpSize;
    cudaCheckError(cudaHostAlloc((void**)&locCmpSize, 
        sizeof(unsigned int) * (totalChunks + 2), cudaHostAllocDefault));
    
    //translated comment
    locCmpSize[0] = 0xDEADBEEF;
    locCmpSize[totalChunks + 1] = 0xCAFEBABE;
    
    unsigned int *h_cmp_offset;
    cudaCheckError(cudaHostAlloc((void**)&h_cmp_offset, 
        sizeof(unsigned int) * (totalChunks + 1), cudaHostAllocDefault));
    h_cmp_offset[0] = 0;
    
    bool *of_rd = new bool[totalChunks + numStreams]();
    of_rd[0] = true;
    
    size_t *chunkElementCounts;
    cudaCheckError(cudaHostAlloc((void**)&chunkElementCounts, 
        sizeof(size_t) * totalChunks, cudaHostAllocDefault));
    
    std::vector<size_t> chunkSizes(totalChunks);
    std::vector<size_t> chunkElementCountsVec(totalChunks);
    
    //translated comment
    const int MAX_EVENTS_PER_TYPE = totalChunks + numStreams;
    
    cudaStream_t *streams = new cudaStream_t[numStreams];
    cudaEvent_t *evSize = new cudaEvent_t[MAX_EVENTS_PER_TYPE];
    cudaEvent_t *evData = new cudaEvent_t[MAX_EVENTS_PER_TYPE];
    Stage *stage = new Stage[numStreams];
    
    for (int i = 0; i < numStreams; ++i) {
        cudaCheckError(cudaStreamCreate(&streams[i]));
        stage[i] = IDLE;
    }
    
    for (int i = 0; i < MAX_EVENTS_PER_TYPE; ++i) {
        cudaCheckError(cudaEventCreateWithFlags(&evSize[i], cudaEventDisableTiming));
        cudaCheckError(cudaEventCreateWithFlags(&evData[i], cudaEventDisableTiming));
    }
    
    //translated comment
    double **d_in = new double*[numStreams];
    unsigned char **d_out = new unsigned char*[numStreams];
    
    for (int i = 0; i < numStreams; ++i) {
        cudaCheckError(cudaMalloc(&d_in[i], chunkSize * sizeof(double)));
        cudaCheckError(cudaMalloc(&d_out[i], chunkSize * sizeof(double) * 1.2));
    }
    
    cudaEventRecord(init_end_event);
    cudaEventSynchronize(init_end_event);
    
    //: stream
    size_t processedEle = 0;
    int active = 0;
    size_t totalCmpSize = 0;
    size_t completedChunks = 0;
    
    cudaDeviceSynchronize();
    cudaEventRecord(global_start_event);
    
    std::vector<int> chunkIDX(numStreams);
    
    while (processedEle < data.nbEle || active > 0) {
        int progress = 0;
        
        for (int s = 0; s < numStreams; ++s) {
            switch (stage[s]) {
                case IDLE:
                    if (processedEle < data.nbEle) {
                        size_t todo = std::min(chunkSize, data.nbEle - processedEle);
                        if(todo == 0) continue;
                        
                        progress = 1;
                        chunkIDX[s] = completedChunks;
                        completedChunks++;
                        
                        chunkElementCounts[chunkIDX[s]] = todo;
                        
                        //translated comment
                        cudaCheckError(cudaMemcpyAsync(
                            d_in[s],
                            data.oriData + processedEle,
                            todo * sizeof(double),
                            cudaMemcpyHostToDevice,
                            streams[s]));
                        
                        //translated comment
                        FalconCompressor::Falcon_compress_no_pack(
                            d_in[s],
                            d_out[s],
                            &locCmpSize[chunkIDX[s] + 1],
                            todo,
                            streams[s]);
                        
                        active += 1;
                        processedEle += todo;
                        stage[s] = SIZE_PENDING;
                        cudaCheckError(cudaEventRecord(evSize[chunkIDX[s]], streams[s]));
                    }
                    break;
                    
                case SIZE_PENDING:
                    if(cudaEventQuery(evSize[chunkIDX[s]]) == cudaSuccess && 
                       of_rd[chunkIDX[s]]) {
                        
                        if (locCmpSize[0] != 0xDEADBEEF || 
                            locCmpSize[totalChunks + 1] != 0xCAFEBABE) {
                            printf("错误:内存保护值被覆写!\n");
                        }
                        
                        progress = 1;
                        int idx = chunkIDX[s];
                        
                        unsigned int compressedBits = locCmpSize[idx + 1];
                        unsigned int compressedBytes = (compressedBits + 7) / 8;
                        
                        //translated comment
                        cudaCheckError(cudaMemcpyAsync(
                            data.cmpBytes + h_cmp_offset[idx],
                            d_out[s],
                            compressedBytes,
                            cudaMemcpyDeviceToHost,
                            streams[s]));
                        
                        h_cmp_offset[idx + 1] = h_cmp_offset[idx] + compressedBytes;
                        of_rd[idx + 1] = true;
                        chunkSizes[idx] = compressedBytes;
                        chunkElementCountsVec[idx] = chunkElementCounts[idx];
                        
                        cudaCheckError(cudaEventRecord(evData[chunkIDX[s]], streams[s]));
                        stage[s] = DATA_PENDING;
                    }
                    break;
                    
                case DATA_PENDING:
                    if (cudaEventQuery(evData[chunkIDX[s]]) == cudaSuccess) {
                        unsigned int compressedBytes = (locCmpSize[chunkIDX[s] + 1] + 7) / 8;
                        totalCmpSize += compressedBytes;
                        progress = 1;
                        stage[s] = IDLE;
                        active -= 1;
                    }
                    break;
            }
        }
        
        //translated comment
        if (!progress) {
            for (int i = 0; i < numStreams; i++) {
                cudaCheckError(cudaStreamSynchronize(streams[i]));
            }
        }
    }
    
    //translated comment
    for (int i = 0; i < std::min(numStreams, (int)totalChunks); i++) {
        cudaCheckError(cudaStreamSynchronize(streams[i]));
        cudaCheckError(cudaEventSynchronize(evData[i]));
    }
    
    cudaEventRecord(global_end_event);
    cudaEventSynchronize(global_end_event);
    
    //translated comment
    float totalTime;
    cudaEventElapsedTime(&totalTime, global_start_event, global_end_event);
    
    //translated comment
    double compressionRatio = totalCmpSize / static_cast<double>(data.nbEle * sizeof(double));
    
    //translated comment
    PipelineAnalysis analysis;
    analysis.compression_ratio = compressionRatio;
    analysis.total_compressed_size = totalCmpSize;
    analysis.total_size = data.nbEle * sizeof(double) / 1024.0 / 1024.0;
    analysis.comp_time = totalTime;
    analysis.comp_throughout = (data.nbEle * sizeof(double) / 1024.0 / 1024.0 / 1024.0) / 
                              (totalTime / 1000.0);
    analysis.chunk_size = chunkSize;
    *data.cmpSize = totalCmpSize;
    
    //translated comment
    for (int i = 0; i < numStreams; i++) {
        cudaCheckError(cudaFree(d_out[i]));
        cudaCheckError(cudaFree(d_in[i]));
    }
    
    //translated comment
    for (int i = 0; i < numStreams; i++) {
        cudaCheckError(cudaStreamDestroy(streams[i]));
    }
    for (int i = 0; i < MAX_EVENTS_PER_TYPE; i++) {
        cudaCheckError(cudaEventDestroy(evSize[i]));
        cudaCheckError(cudaEventDestroy(evData[i]));
    }
    
    //translated comment
    cudaCheckError(cudaFreeHost(locCmpSize));
    cudaCheckError(cudaFreeHost(h_cmp_offset));
    cudaCheckError(cudaFreeHost(chunkElementCounts));
    delete[] of_rd;
    
    //translated comment
    cudaCheckError(cudaEventDestroy(global_start_event));
    cudaCheckError(cudaEventDestroy(global_end_event));
    cudaCheckError(cudaEventDestroy(init_start_event));
    cudaCheckError(cudaEventDestroy(init_end_event));
    
    //translated comment
    delete[] streams;
    delete[] evSize;
    delete[] evData;
    delete[] stage;
    delete[] d_in;
    delete[] d_out;
    
    //translated comment
    CompressionResult result;
    result.analysis = analysis;
    result.chunkSizes = std::move(chunkSizes);
    result.chunkElementCounts = std::move(chunkElementCountsVec);
    result.totalChunks = completedChunks;
    
    return result;
}

// SPARE---------------------------------------------------------------------


CompressionResult FalconPipeline::executeCompressionPipelineSpare(
    ProcessedData &data,
    size_t chunkSize) {
    
    return executeCompressionPipelineImpl_Spare(data, chunkSize, NUM_STREAMS);
}

//translated comment
CompressionResult FalconPipeline::executeCompressionPipelineImpl_Spare(
    ProcessedData &data,
    size_t chunkSize,
    int numStreams) {
    
    cudaDeviceSynchronize();
    
    //translated comment
    cudaEvent_t global_start_event, global_end_event;
    cudaEventCreate(&global_start_event);
    cudaEventCreate(&global_end_event);
    
    cudaEvent_t init_start_event, init_end_event;
    cudaEventCreate(&init_start_event);
    cudaEventCreate(&init_end_event);
    
    //translated comment
    cudaDeviceSynchronize();
    cudaEventRecord(init_start_event);
    
    //chunk
    size_t totalChunks = (data.nbEle + chunkSize - 1) / chunkSize;
    if(totalChunks == 1) {
        chunkSize = data.nbEle;
    }
    
    printf("totalChunks: %zu\n", totalChunks);
    
    //translated comment
    unsigned int *locCmpSize;
    cudaCheckError(cudaHostAlloc((void**)&locCmpSize, 
        sizeof(unsigned int) * (totalChunks + 2), cudaHostAllocDefault));
    
    //translated comment
    locCmpSize[0] = 0xDEADBEEF;
    locCmpSize[totalChunks + 1] = 0xCAFEBABE;
    
    unsigned int *h_cmp_offset;
    cudaCheckError(cudaHostAlloc((void**)&h_cmp_offset, 
        sizeof(unsigned int) * (totalChunks + 1), cudaHostAllocDefault));
    h_cmp_offset[0] = 0;
    
    bool *of_rd = new bool[totalChunks + numStreams]();
    of_rd[0] = true;
    
    size_t *chunkElementCounts;
    cudaCheckError(cudaHostAlloc((void**)&chunkElementCounts, 
        sizeof(size_t) * totalChunks, cudaHostAllocDefault));
    
    std::vector<size_t> chunkSizes(totalChunks);
    std::vector<size_t> chunkElementCountsVec(totalChunks);
    
    //translated comment
    const int MAX_EVENTS_PER_TYPE = totalChunks + numStreams;
    
    cudaStream_t *streams = new cudaStream_t[numStreams];
    cudaEvent_t *evSize = new cudaEvent_t[MAX_EVENTS_PER_TYPE];
    cudaEvent_t *evData = new cudaEvent_t[MAX_EVENTS_PER_TYPE];
    Stage *stage = new Stage[numStreams];
    
    for (int i = 0; i < numStreams; ++i) {
        cudaCheckError(cudaStreamCreate(&streams[i]));
        stage[i] = IDLE;
    }
    
    for (int i = 0; i < MAX_EVENTS_PER_TYPE; ++i) {
        cudaCheckError(cudaEventCreateWithFlags(&evSize[i], cudaEventDisableTiming));
        cudaCheckError(cudaEventCreateWithFlags(&evData[i], cudaEventDisableTiming));
    }
    
    //translated comment
    double **d_in = new double*[numStreams];
    unsigned char **d_out = new unsigned char*[numStreams];
    
    for (int i = 0; i < numStreams; ++i) {
        cudaCheckError(cudaMalloc(&d_in[i], chunkSize * sizeof(double)));
        cudaCheckError(cudaMalloc(&d_out[i], chunkSize * sizeof(double)));
    }
    
    cudaEventRecord(init_end_event);
    cudaEventSynchronize(init_end_event);
    
    //: stream
    size_t processedEle = 0;
    int active = 0;
    size_t totalCmpSize = 0;
    size_t completedChunks = 0;
    
    cudaDeviceSynchronize();
    cudaEventRecord(global_start_event);
    
    std::vector<int> chunkIDX(numStreams);
    
    while (processedEle < data.nbEle || active > 0) {
        int progress = 0;
        
        for (int s = 0; s < numStreams; ++s) {
            switch (stage[s]) {
                case IDLE:
                    if (processedEle < data.nbEle) {
                        size_t todo = std::min(chunkSize, data.nbEle - processedEle);
                        if(todo == 0) continue;
                        
                        progress = 1;
                        chunkIDX[s] = completedChunks;
                        completedChunks++;
                        
                        chunkElementCounts[chunkIDX[s]] = todo;
                        
                        //translated comment
                        cudaCheckError(cudaMemcpyAsync(
                            d_in[s],
                            data.oriData + processedEle,
                            todo * sizeof(double),
                            cudaMemcpyHostToDevice,
                            streams[s]));
                        
                        //translated comment
                        FalconCompressor::Falcon_compress_spare(
                            d_in[s],
                            d_out[s],
                            &locCmpSize[chunkIDX[s] + 1],
                            todo,
                            streams[s]);
                        
                        active += 1;
                        processedEle += todo;
                        stage[s] = SIZE_PENDING;
                        cudaCheckError(cudaEventRecord(evSize[chunkIDX[s]], streams[s]));
                    }
                    break;
                    
                case SIZE_PENDING:
                    if(cudaEventQuery(evSize[chunkIDX[s]]) == cudaSuccess && 
                       of_rd[chunkIDX[s]]) {
                        
                        if (locCmpSize[0] != 0xDEADBEEF || 
                            locCmpSize[totalChunks + 1] != 0xCAFEBABE) {
                            printf("错误:内存保护值被覆写!\n");
                        }
                        
                        progress = 1;
                        int idx = chunkIDX[s];
                        
                        unsigned int compressedBits = locCmpSize[idx + 1];
                        unsigned int compressedBytes = (compressedBits + 7) / 8;
                        
                        //translated comment
                        cudaCheckError(cudaMemcpyAsync(
                            data.cmpBytes + h_cmp_offset[idx],
                            d_out[s],
                            compressedBytes,
                            cudaMemcpyDeviceToHost,
                            streams[s]));
                        
                        h_cmp_offset[idx + 1] = h_cmp_offset[idx] + compressedBytes;
                        of_rd[idx + 1] = true;
                        chunkSizes[idx] = compressedBytes;
                        chunkElementCountsVec[idx] = chunkElementCounts[idx];
                        
                        cudaCheckError(cudaEventRecord(evData[chunkIDX[s]], streams[s]));
                        stage[s] = DATA_PENDING;
                    }
                    break;
                    
                case DATA_PENDING:
                    if (cudaEventQuery(evData[chunkIDX[s]]) == cudaSuccess) {
                        unsigned int compressedBytes = (locCmpSize[chunkIDX[s] + 1] + 7) / 8;
                        totalCmpSize += compressedBytes;
                        progress = 1;
                        stage[s] = IDLE;
                        active -= 1;
                    }
                    break;
            }
        }
        
        //translated comment
        if (!progress) {
            for (int i = 0; i < numStreams; i++) {
                cudaCheckError(cudaStreamSynchronize(streams[i]));
            }
        }
    }
    
    //translated comment
    for (int i = 0; i < std::min(numStreams, (int)totalChunks); i++) {
        cudaCheckError(cudaStreamSynchronize(streams[i]));
        cudaCheckError(cudaEventSynchronize(evData[i]));
    }
    
    cudaEventRecord(global_end_event);
    cudaEventSynchronize(global_end_event);
    
    //translated comment
    float totalTime;
    cudaEventElapsedTime(&totalTime, global_start_event, global_end_event);
    
    //translated comment
    double compressionRatio = totalCmpSize / static_cast<double>(data.nbEle * sizeof(double));
    
    //translated comment
    PipelineAnalysis analysis;
    analysis.compression_ratio = compressionRatio;
    analysis.total_compressed_size = totalCmpSize;
    analysis.total_size = data.nbEle * sizeof(double) / 1024.0 / 1024.0;
    analysis.comp_time = totalTime;
    analysis.comp_throughout = (data.nbEle * sizeof(double) / 1024.0 / 1024.0 / 1024.0) / 
                              (totalTime / 1000.0);
    analysis.chunk_size = chunkSize;
    *data.cmpSize = totalCmpSize;
    
    //translated comment
    for (int i = 0; i < numStreams; i++) {
        cudaCheckError(cudaFree(d_out[i]));
        cudaCheckError(cudaFree(d_in[i]));
    }
    
    //translated comment
    for (int i = 0; i < numStreams; i++) {
        cudaCheckError(cudaStreamDestroy(streams[i]));
    }
    for (int i = 0; i < MAX_EVENTS_PER_TYPE; i++) {
        cudaCheckError(cudaEventDestroy(evSize[i]));
        cudaCheckError(cudaEventDestroy(evData[i]));
    }
    
    //translated comment
    cudaCheckError(cudaFreeHost(locCmpSize));
    cudaCheckError(cudaFreeHost(h_cmp_offset));
    cudaCheckError(cudaFreeHost(chunkElementCounts));
    delete[] of_rd;
    
    //translated comment
    cudaCheckError(cudaEventDestroy(global_start_event));
    cudaCheckError(cudaEventDestroy(global_end_event));
    cudaCheckError(cudaEventDestroy(init_start_event));
    cudaCheckError(cudaEventDestroy(init_end_event));
    
    //translated comment
    delete[] streams;
    delete[] evSize;
    delete[] evData;
    delete[] stage;
    delete[] d_in;
    delete[] d_out;
    
    //translated comment
    CompressionResult result;
    result.analysis = analysis;
    result.chunkSizes = std::move(chunkSizes);
    result.chunkElementCounts = std::move(chunkElementCountsVec);
    result.totalChunks = completedChunks;
    
    return result;
}

// BR---------------------------------------------------------------------


CompressionResult FalconPipeline::executeCompressionPipelineBr(
    ProcessedData &data,
    size_t chunkSize) {
    
    return executeCompressionPipelineImpl_Br(data, chunkSize, NUM_STREAMS);
}

//translated comment
CompressionResult FalconPipeline::executeCompressionPipelineImpl_Br(
    ProcessedData &data,
    size_t chunkSize,
    int numStreams) {
    
    cudaDeviceSynchronize();
    
    //translated comment
    cudaEvent_t global_start_event, global_end_event;
    cudaEventCreate(&global_start_event);
    cudaEventCreate(&global_end_event);
    
    cudaEvent_t init_start_event, init_end_event;
    cudaEventCreate(&init_start_event);
    cudaEventCreate(&init_end_event);
    
    //translated comment
    cudaDeviceSynchronize();
    cudaEventRecord(init_start_event);
    
    //chunk
    size_t totalChunks = (data.nbEle + chunkSize - 1) / chunkSize;
    if(totalChunks == 1) {
        chunkSize = data.nbEle;
    }
    
    printf("totalChunks: %zu\n", totalChunks);
    
    //translated comment
    unsigned int *locCmpSize;
    cudaCheckError(cudaHostAlloc((void**)&locCmpSize, 
        sizeof(unsigned int) * (totalChunks + 2), cudaHostAllocDefault));
    
    //translated comment
    locCmpSize[0] = 0xDEADBEEF;
    locCmpSize[totalChunks + 1] = 0xCAFEBABE;
    
    unsigned int *h_cmp_offset;
    cudaCheckError(cudaHostAlloc((void**)&h_cmp_offset, 
        sizeof(unsigned int) * (totalChunks + 1), cudaHostAllocDefault));
    h_cmp_offset[0] = 0;
    
    bool *of_rd = new bool[totalChunks + numStreams]();
    of_rd[0] = true;
    
    size_t *chunkElementCounts;
    cudaCheckError(cudaHostAlloc((void**)&chunkElementCounts, 
        sizeof(size_t) * totalChunks, cudaHostAllocDefault));
    
    std::vector<size_t> chunkSizes(totalChunks);
    std::vector<size_t> chunkElementCountsVec(totalChunks);
    
    //translated comment
    const int MAX_EVENTS_PER_TYPE = totalChunks + numStreams;
    
    cudaStream_t *streams = new cudaStream_t[numStreams];
    cudaEvent_t *evSize = new cudaEvent_t[MAX_EVENTS_PER_TYPE];
    cudaEvent_t *evData = new cudaEvent_t[MAX_EVENTS_PER_TYPE];
    Stage *stage = new Stage[numStreams];
    
    for (int i = 0; i < numStreams; ++i) {
        cudaCheckError(cudaStreamCreate(&streams[i]));
        stage[i] = IDLE;
    }
    
    for (int i = 0; i < MAX_EVENTS_PER_TYPE; ++i) {
        cudaCheckError(cudaEventCreateWithFlags(&evSize[i], cudaEventDisableTiming));
        cudaCheckError(cudaEventCreateWithFlags(&evData[i], cudaEventDisableTiming));
    }
    
    //translated comment
    double **d_in = new double*[numStreams];
    unsigned char **d_out = new unsigned char*[numStreams];
    
    for (int i = 0; i < numStreams; ++i) {
        cudaCheckError(cudaMalloc(&d_in[i], chunkSize * sizeof(double)));
        cudaCheckError(cudaMalloc(&d_out[i], chunkSize * sizeof(double)));
    }
    
    cudaEventRecord(init_end_event);
    cudaEventSynchronize(init_end_event);
    
    //: stream
    size_t processedEle = 0;
    int active = 0;
    size_t totalCmpSize = 0;
    size_t completedChunks = 0;
    
    cudaDeviceSynchronize();
    cudaEventRecord(global_start_event);
    
    std::vector<int> chunkIDX(numStreams);
    
    while (processedEle < data.nbEle || active > 0) {
        int progress = 0;
        
        for (int s = 0; s < numStreams; ++s) {
            switch (stage[s]) {
                case IDLE:
                    if (processedEle < data.nbEle) {
                        size_t todo = std::min(chunkSize, data.nbEle - processedEle);
                        if(todo == 0) continue;
                        
                        progress = 1;
                        chunkIDX[s] = completedChunks;
                        completedChunks++;
                        
                        chunkElementCounts[chunkIDX[s]] = todo;
                        
                        //translated comment
                        cudaCheckError(cudaMemcpyAsync(
                            d_in[s],
                            data.oriData + processedEle,
                            todo * sizeof(double),
                            cudaMemcpyHostToDevice,
                            streams[s]));
                        
                        //translated comment
                        FalconCompressor::Falcon_compress_br(
                            d_in[s],
                            d_out[s],
                            &locCmpSize[chunkIDX[s] + 1],
                            todo,
                            streams[s]);
                        
                        active += 1;
                        processedEle += todo;
                        stage[s] = SIZE_PENDING;
                        cudaCheckError(cudaEventRecord(evSize[chunkIDX[s]], streams[s]));
                    }
                    break;
                    
                case SIZE_PENDING:
                    if(cudaEventQuery(evSize[chunkIDX[s]]) == cudaSuccess && 
                       of_rd[chunkIDX[s]]) {
                        
                        if (locCmpSize[0] != 0xDEADBEEF || 
                            locCmpSize[totalChunks + 1] != 0xCAFEBABE) {
                            printf("错误:内存保护值被覆写!\n");
                        }
                        
                        progress = 1;
                        int idx = chunkIDX[s];
                        
                        unsigned int compressedBits = locCmpSize[idx + 1];
                        unsigned int compressedBytes = (compressedBits + 7) / 8;
                        
                        //translated comment
                        cudaCheckError(cudaMemcpyAsync(
                            data.cmpBytes + h_cmp_offset[idx],
                            d_out[s],
                            compressedBytes,
                            cudaMemcpyDeviceToHost,
                            streams[s]));
                        
                        h_cmp_offset[idx + 1] = h_cmp_offset[idx] + compressedBytes;
                        of_rd[idx + 1] = true;
                        chunkSizes[idx] = compressedBytes;
                        chunkElementCountsVec[idx] = chunkElementCounts[idx];
                        
                        cudaCheckError(cudaEventRecord(evData[chunkIDX[s]], streams[s]));
                        stage[s] = DATA_PENDING;
                    }
                    break;
                    
                case DATA_PENDING:
                    if (cudaEventQuery(evData[chunkIDX[s]]) == cudaSuccess) {
                        unsigned int compressedBytes = (locCmpSize[chunkIDX[s] + 1] + 7) / 8;
                        totalCmpSize += compressedBytes;
                        progress = 1;
                        stage[s] = IDLE;
                        active -= 1;
                    }
                    break;
            }
        }
        
        //translated comment
        if (!progress) {
            for (int i = 0; i < numStreams; i++) {
                cudaCheckError(cudaStreamSynchronize(streams[i]));
            }
        }
    }
    
    //translated comment
    for (int i = 0; i < std::min(numStreams, (int)totalChunks); i++) {
        cudaCheckError(cudaStreamSynchronize(streams[i]));
        cudaCheckError(cudaEventSynchronize(evData[i]));
    }
    
    cudaEventRecord(global_end_event);
    cudaEventSynchronize(global_end_event);
    
    //translated comment
    float totalTime;
    cudaEventElapsedTime(&totalTime, global_start_event, global_end_event);
    
    //translated comment
    double compressionRatio = totalCmpSize / static_cast<double>(data.nbEle * sizeof(double));
    
    //translated comment
    PipelineAnalysis analysis;
    analysis.compression_ratio = compressionRatio;
    analysis.total_compressed_size = totalCmpSize;
    analysis.total_size = data.nbEle * sizeof(double) / 1024.0 / 1024.0;
    analysis.comp_time = totalTime;
    analysis.comp_throughout = (data.nbEle * sizeof(double) / 1024.0 / 1024.0 / 1024.0) / 
                              (totalTime / 1000.0);
    analysis.chunk_size = chunkSize;
    *data.cmpSize = totalCmpSize;
    
    //translated comment
    for (int i = 0; i < numStreams; i++) {
        cudaCheckError(cudaFree(d_out[i]));
        cudaCheckError(cudaFree(d_in[i]));
    }
    
    //translated comment
    for (int i = 0; i < numStreams; i++) {
        cudaCheckError(cudaStreamDestroy(streams[i]));
    }
    for (int i = 0; i < MAX_EVENTS_PER_TYPE; i++) {
        cudaCheckError(cudaEventDestroy(evSize[i]));
        cudaCheckError(cudaEventDestroy(evData[i]));
    }
    
    //translated comment
    cudaCheckError(cudaFreeHost(locCmpSize));
    cudaCheckError(cudaFreeHost(h_cmp_offset));
    cudaCheckError(cudaFreeHost(chunkElementCounts));
    delete[] of_rd;
    
    //translated comment
    cudaCheckError(cudaEventDestroy(global_start_event));
    cudaCheckError(cudaEventDestroy(global_end_event));
    cudaCheckError(cudaEventDestroy(init_start_event));
    cudaCheckError(cudaEventDestroy(init_end_event));
    
    //translated comment
    delete[] streams;
    delete[] evSize;
    delete[] evData;
    delete[] stage;
    delete[] d_in;
    delete[] d_out;
    
    //translated comment
    CompressionResult result;
    result.analysis = analysis;
    result.chunkSizes = std::move(chunkSizes);
    result.chunkElementCounts = std::move(chunkElementCountsVec);
    result.totalChunks = completedChunks;
    
    return result;
}

//translated comment

//translated comment

CompressionResult FalconPipeline::executeCompressionPipelineBlock(
    ProcessedData &data,
    size_t chunkSize) 
{
    cudaEvent_t global_start_event, global_end_event; //translated comment
    cudaDeviceSynchronize();
    //translated comment
    cudaEventCreate(&global_start_event);
    cudaEventCreate(&global_end_event);
    //translated comment
    cudaEvent_t init_start_event,init_end_event;
    cudaEventCreate(&init_start_event);
    cudaEventCreate(&init_end_event);

    //translated comment
    cudaDeviceSynchronize();//translated comment
    cudaEventRecord(init_start_event);

    //chunk
    size_t totalChunks = (data.nbEle + chunkSize - 1) / chunkSize;
    if(totalChunks==1)
    // if(1)
    {
        chunkSize=data.nbEle;
    }
    //translated comment
    //translated comment
    unsigned int *locCmpSize;
    cudaCheckError(cudaHostAlloc((void**)&locCmpSize, 
    sizeof(unsigned int) * (totalChunks + 2), cudaHostAllocDefault));  // +2 for guards

    //translated comment
    locCmpSize[0] = 0xDEADBEEF;  //translated comment
    locCmpSize[totalChunks + 1] = 0xCAFEBABE;  //translated comment


    
    unsigned int *h_cmp_offset;
    cudaCheckError(cudaHostAlloc((void**)&h_cmp_offset, sizeof(unsigned int) * (totalChunks + 1), cudaHostAllocDefault));
    //translated comment
    h_cmp_offset[0] = 0;

    //： chunk
    size_t *chunkElementCounts;
    cudaCheckError(cudaHostAlloc((void**)&chunkElementCounts, sizeof(size_t) * totalChunks, cudaHostAllocDefault));

    //chunk

    std::vector<size_t> chunkSizes(totalChunks);
    std::vector<size_t> chunkElementCountsVec(totalChunks);

    //translated comment
    const int MAX_EVENTS_PER_TYPE = totalChunks + NUM_STREAMS; //translated comment
    
    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t kernal_start[MAX_EVENTS_PER_TYPE]; //translated comment
    cudaEvent_t evSize[MAX_EVENTS_PER_TYPE]; //size
    cudaEvent_t evData[MAX_EVENTS_PER_TYPE]; //translated comment
    Stage stage[NUM_STREAMS]; //translated comment

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaCheckError(cudaStreamCreate(&streams[i]));
        stage[i] = IDLE;
    }

    for (int i = 0; i < MAX_EVENTS_PER_TYPE; ++i) {
        cudaCheckError(cudaEventCreateWithFlags(&kernal_start[i], cudaEventDisableTiming));
        cudaCheckError(cudaEventCreateWithFlags(&evSize[i], cudaEventDisableTiming));
        cudaCheckError(cudaEventCreateWithFlags(&evData[i], cudaEventDisableTiming));
    }


    //translated comment
    double *d_in[NUM_STREAMS];
    unsigned char *d_out[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaCheckError(cudaMalloc(&d_in[i], chunkSize * sizeof(double)));
        cudaCheckError(cudaMalloc(&d_out[i], chunkSize * sizeof(double)));
    }


    cudaEventRecord(init_end_event);
    //translated comment
    cudaEventSynchronize(init_end_event);

    //---------- ： stream ----------
    size_t processedEle = 0; //translated comment
    int active = 0;
    size_t totalCmpSize = 0; //translated comment
    size_t completedChunks = 0; //chunk
    //translated comment
    cudaDeviceSynchronize();//translated comment
    cudaEventRecord(global_start_event);

    std::vector<int> chunkIDX(NUM_STREAMS);//compINFO
    while (processedEle < data.nbEle || active > 0) {
        int progress=0;
        for (int s = 0; s < NUM_STREAMS; ++s) {
            switch (stage[s]) {
                case IDLE:
                    if (processedEle < data.nbEle) {
                        //translated comment
                        size_t todo = std::min(chunkSize, data.nbEle - processedEle);
                        if(todo==0)
                        {
                            continue;
                        }
                        progress=1;
                        chunkIDX[s]=completedChunks;//chunks
                        completedChunks++;
                        //translated comment
                        chunkElementCounts[chunkIDX[s]] = todo;

                        //translated comment
                        cudaCheckError(cudaMemcpyAsync(
                            d_in[s],
                            data.oriData + processedEle,
                            todo * sizeof(double), //translated comment
                            cudaMemcpyHostToDevice,
                            streams[s]));
                        //translated comment
                        FalconCompressor::Falcon_compress_stream(
                            d_in[s],
                            d_out[s],
                            &locCmpSize[chunkIDX[s] + 1],  //translated comment
                            todo,
                            streams[s]);

                        //translated comment
                        active += 1;
                        processedEle += todo;
                          
                        // cudaCheckError(cudaSynchronize());
                        cudaDeviceSynchronize();

                        int idx=chunkIDX[s];
                        // printf("idx: %d,stream: %d SIZE\n",idx,s);
                        unsigned int compressedBits = locCmpSize[idx+1];//translated comment
                        unsigned int compressedBytes = (compressedBits + 7) / 8;

                        //translated comment
                        cudaCheckError(cudaMemcpyAsync(
                            data.cmpBytes + h_cmp_offset[idx],
                            d_out[s],
                            compressedBytes,//translated comment
                            cudaMemcpyDeviceToHost,
                            streams[s]));

                        //translated comment
                        h_cmp_offset[idx + 1] = h_cmp_offset[idx] + compressedBytes;
                        chunkSizes[idx] = compressedBytes;
                        chunkElementCountsVec[idx] = chunkElementCounts[idx];

                        totalCmpSize += compressedBytes;
                        active -= 1;
                    }
                    break;

            }
        }
        //translated comment
        if (!progress) {
            for (int i = 0; i < NUM_STREAMS; i++) {
                cudaCheckError(cudaStreamSynchronize(streams[i]));
            }
        }
    }

    //translated comment
    for (int i = 0; i < (NUM_STREAMS>totalChunks?totalChunks:NUM_STREAMS); i++) {
        cudaCheckError(cudaStreamSynchronize(streams[i]));
        // cudaCheckError(cudaEventSynchronize(evData[i]));
    }
    //translated comment
    // unsigned char* tempBuffer = new unsigned char[totalCmpSize];

    // size_t off = 0;
    // for(int i = 0; i < totalChunks; i++)
    // {
    //     unsigned int compressedBytes = (locCmpSize[i + 1] + 7) / 8;
        
    //     memcpy(
    //         tempBuffer + off,
    //         data.cmpBytes + i * chunkSize * sizeof(double),
    //         compressedBytes);
        
    //     off += compressedBytes;
    // }
    // if(totalCmpSize!= off)
    // {
    //     printf("wrong\n");
    // }
    //translated comment
    // memcpy(data.cmpBytes, tempBuffer, totalCmpSize);
    // delete[] tempBuffer;
    //translated comment
    cudaEventRecord(global_end_event);

    //translated comment
    cudaEventSynchronize(global_end_event);
  
    //translated comment
    float totalTime;
    cudaEventElapsedTime(&totalTime, global_start_event, global_end_event);
    float initTime;
    cudaEventElapsedTime(&initTime, init_start_event, init_end_event);
    //translated comment
    double compressionRatio =totalCmpSize/static_cast<double>(data.nbEle * sizeof(double)) ;

    //translated comment
    PipelineAnalysis analysis;
    analysis.compression_ratio = compressionRatio;
    analysis.total_compressed_size = totalCmpSize;
    analysis.total_size = data.nbEle * sizeof(double)/1024/1024;
    analysis.comp_time = totalTime;
    analysis.comp_throughout=(data.nbEle * sizeof(double) / 1024.0 / 1024.0 / 1024.0) / (totalTime / 1000.0);
    analysis.chunk_size = chunkSize;
    *data.cmpSize=totalCmpSize;

    //translated comment
    //translated comment
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaCheckError(cudaFree(d_out[i]));
        cudaCheckError(cudaFree(d_in[i]));
    }

    //translated comment
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaCheckError(cudaStreamDestroy(streams[i]));
    }
    for (int i = 0; i < MAX_EVENTS_PER_TYPE; i++) {
        cudaCheckError(cudaEventDestroy(evSize[i]));
        cudaCheckError(cudaEventDestroy(evData[i]));
    }

    //translated comment
    cudaCheckError(cudaFreeHost(locCmpSize));
    cudaCheckError(cudaFreeHost(h_cmp_offset));
    cudaCheckError(cudaFreeHost(chunkElementCounts));

    //translated comment
    cudaCheckError(cudaEventDestroy(global_start_event));
    cudaCheckError(cudaEventDestroy(global_end_event));

    //translated comment
    CompressionResult result;
    result.analysis = analysis;
    result.chunkSizes = std::move(chunkSizes);
    result.chunkElementCounts = std::move(chunkElementCountsVec);
    result.totalChunks = completedChunks;
    return result;
}

//translated comment

CompressionResult FalconPipeline::executeCompressionPipelineNoBlock(
    ProcessedData &data,
    size_t chunkSize) 
{
    cudaEvent_t global_start_event, global_end_event; //translated comment
    cudaDeviceSynchronize();
    //translated comment
    cudaEventCreate(&global_start_event);
    cudaEventCreate(&global_end_event);
    //translated comment
    cudaEvent_t init_start_event,init_end_event;
    cudaEventCreate(&init_start_event);
    cudaEventCreate(&init_end_event);

    //translated comment
    cudaDeviceSynchronize();//translated comment
    cudaEventRecord(init_start_event);

    //chunk
    size_t totalChunks = (data.nbEle + chunkSize - 1) / chunkSize;
    if(totalChunks==1)
    {
        chunkSize=data.nbEle;
    }
    //translated comment
    //translated comment
    unsigned int *locCmpSize;
    cudaCheckError(cudaHostAlloc((void**)&locCmpSize, 
    sizeof(unsigned int) * (totalChunks + 2), cudaHostAllocDefault));  // +2 for guards

    //translated comment
    locCmpSize[0] = 0xDEADBEEF;  //translated comment
    locCmpSize[totalChunks + 1] = 0xCAFEBABE;  //translated comment


    
    unsigned int *h_cmp_offset;
    cudaCheckError(cudaHostAlloc((void**)&h_cmp_offset, sizeof(unsigned int) * (totalChunks + 1), cudaHostAllocDefault));
    //translated comment
    h_cmp_offset[0] = 0;
    //： chunk
    size_t *chunkElementCounts;
    cudaCheckError(cudaHostAlloc((void**)&chunkElementCounts, sizeof(size_t) * totalChunks, cudaHostAllocDefault));

    //chunk

    std::vector<size_t> chunkSizes(totalChunks);
    std::vector<size_t> chunkElementCountsVec(totalChunks);

    //translated comment
    const int MAX_EVENTS_PER_TYPE = totalChunks + NUM_STREAMS; //translated comment
    
    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t kernal_start[MAX_EVENTS_PER_TYPE]; //translated comment
    cudaEvent_t evSize[MAX_EVENTS_PER_TYPE]; //size
    cudaEvent_t evData[MAX_EVENTS_PER_TYPE]; //translated comment
    Stage stage[NUM_STREAMS]; //translated comment

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaCheckError(cudaStreamCreate(&streams[i]));
        stage[i] = IDLE;
    }

    for (int i = 0; i < MAX_EVENTS_PER_TYPE; ++i) {
        cudaCheckError(cudaEventCreateWithFlags(&kernal_start[i], cudaEventDisableTiming));
        cudaCheckError(cudaEventCreateWithFlags(&evSize[i], cudaEventDisableTiming));
        cudaCheckError(cudaEventCreateWithFlags(&evData[i], cudaEventDisableTiming));
    }


    //translated comment
    double *d_in[NUM_STREAMS];
    unsigned char *d_out[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaCheckError(cudaMalloc(&d_in[i], chunkSize * sizeof(double)));
        cudaCheckError(cudaMalloc(&d_out[i], chunkSize * sizeof(double)));
    }


    cudaEventRecord(init_end_event);
    //translated comment
    cudaEventSynchronize(init_end_event);

    //---------- ： stream ----------
    size_t processedEle = 0; //translated comment
    int active = 0;
    size_t totalCmpSize = 0; //translated comment
    size_t completedChunks = 0; //chunk
    //translated comment
    cudaDeviceSynchronize();//translated comment
    cudaEventRecord(global_start_event);

    std::vector<int> chunkIDX(NUM_STREAMS);//compINFO
    while (processedEle < data.nbEle || active > 0) {
        int progress=0;
        for (int s = 0; s < NUM_STREAMS; ++s) {
            switch (stage[s]) {
                case IDLE:
                    if (processedEle < data.nbEle) {
                        //translated comment
                        size_t todo = std::min(chunkSize, data.nbEle - processedEle);
                        if(todo==0)
                        {
                            continue;
                        }
                        progress=1;
                        chunkIDX[s]=completedChunks;//chunks
                        completedChunks++;
                        //translated comment
                        chunkElementCounts[chunkIDX[s]] = todo;

                        //translated comment
                        cudaCheckError(cudaMemcpyAsync(
                            d_in[s],
                            data.oriData + processedEle,
                            todo * sizeof(double), //translated comment
                            cudaMemcpyHostToDevice,
                            streams[s]));
                        //translated comment
                        FalconCompressor::Falcon_compress_stream(
                            d_in[s],
                            d_out[s],
                            &locCmpSize[chunkIDX[s] + 1],  //translated comment
                            todo,
                            streams[s]);

                        //translated comment
                        active += 1;
                        processedEle += todo;
                          
                        stage[s] = SIZE_PENDING;
                        cudaCheckError(cudaEventRecord(evSize[chunkIDX[s]], streams[s]));
                    }
                    break;
                case SIZE_PENDING:
                    //translated comment
                    if(cudaEventQuery(evSize[chunkIDX[s]]) == cudaSuccess ) {
                        if (locCmpSize[0] != 0xDEADBEEF || locCmpSize[totalChunks + 1] != 0xCAFEBABE) {
                            printf("错误：内存保护值被覆写！\n");
                        }
                        progress=1;
                        //translated comment
                        int idx=chunkIDX[s];
                        // printf("idx: %d,stream: %d SIZE\n",idx,s);
                        unsigned int compressedBits = locCmpSize[idx+1];//translated comment
                        unsigned int compressedBytes = (compressedBits + 7) / 8;

                        //translated comment
                        cudaCheckError(cudaMemcpyAsync(
                            data.cmpBytes + idx * chunkSize * sizeof(double),
                            d_out[s],
                            chunkElementCounts[chunkIDX[s]]  * sizeof(double),//translated comment
                            cudaMemcpyDeviceToHost,
                            streams[s]));

                        //translated comment
                        chunkSizes[idx] = compressedBytes;
                        chunkElementCountsVec[idx] = chunkElementCounts[idx];
                        cudaCheckError(cudaEventRecord(evData[chunkIDX[s]], streams[s]));
                        stage[s] = DATA_PENDING;
                    }
                    break;

                case DATA_PENDING:
                    if (cudaEventQuery(evData[chunkIDX[s]]) == cudaSuccess) {
                        //translated comment
                        unsigned int compressedBytes = (locCmpSize[chunkIDX[s]+1] + 7) / 8;//translated comment
                        totalCmpSize += compressedBytes;
                        progress=1;
                        stage[s] = IDLE;
                        active -= 1;
                    }
                    break;
            }
        }
        //translated comment
        if (!progress) {
            for (int i = 0; i < NUM_STREAMS; i++) {
                cudaCheckError(cudaStreamSynchronize(streams[i]));
            }
        }
    }

    //translated comment
    for (int i = 0; i < (NUM_STREAMS>totalChunks?totalChunks:NUM_STREAMS); i++) {
        cudaCheckError(cudaStreamSynchronize(streams[i]));
        cudaCheckError(cudaEventSynchronize(evData[i]));
    }

    //translated comment
    unsigned char* tempBuffer = new unsigned char[totalCmpSize];

    size_t off = 0;
    for(int i = 0; i < totalChunks; i++)
    {
        unsigned int compressedBytes = (locCmpSize[i + 1] + 7) / 8;
        
        memcpy(
            tempBuffer + off,
            data.cmpBytes + i * chunkSize * sizeof(double),
            compressedBytes);
        
        off += compressedBytes;
    }
    if(totalCmpSize!= off)
    {
        printf("wrong\n");
    }
    //translated comment
    memcpy(data.cmpBytes, tempBuffer, totalCmpSize);
    delete[] tempBuffer;


    //translated comment
    cudaEventRecord(global_end_event);

    //translated comment
    cudaEventSynchronize(global_end_event);

    //translated comment
    float totalTime;
    cudaEventElapsedTime(&totalTime, global_start_event, global_end_event);
    float initTime;
    cudaEventElapsedTime(&initTime, init_start_event, init_end_event);
    //translated comment
    double compressionRatio =totalCmpSize/static_cast<double>(data.nbEle * sizeof(double)) ;

    //translated comment
    PipelineAnalysis analysis;
    analysis.compression_ratio = compressionRatio;
    analysis.total_compressed_size = totalCmpSize;
    analysis.total_size = data.nbEle * sizeof(double)/1024/1024;
    analysis.comp_time = totalTime;
    analysis.comp_throughout=(data.nbEle * sizeof(double) / 1024.0 / 1024.0 / 1024.0) / (totalTime / 1000.0);
    analysis.chunk_size = chunkSize;
    *data.cmpSize=totalCmpSize;

    //translated comment
    //translated comment
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaCheckError(cudaFree(d_out[i]));
        cudaCheckError(cudaFree(d_in[i]));
    }

    //translated comment
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaCheckError(cudaStreamDestroy(streams[i]));
    }
    for (int i = 0; i < MAX_EVENTS_PER_TYPE; i++) {
        cudaCheckError(cudaEventDestroy(evSize[i]));
        cudaCheckError(cudaEventDestroy(evData[i]));
    }

    //translated comment
    cudaCheckError(cudaFreeHost(locCmpSize));
    cudaCheckError(cudaFreeHost(h_cmp_offset));
    cudaCheckError(cudaFreeHost(chunkElementCounts));

    //translated comment
    cudaCheckError(cudaEventDestroy(global_start_event));
    cudaCheckError(cudaEventDestroy(global_end_event));

    //translated comment
    CompressionResult result;
    result.analysis = analysis;
    result.chunkSizes = std::move(chunkSizes);
    result.chunkElementCounts = std::move(chunkElementCountsVec);
    result.totalChunks = completedChunks;
    // result.cmpInfo=tmp;
    return result;
}
