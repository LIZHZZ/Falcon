// Falcon_float_pipeline.cu
// 32-bit floating-point pipeline compression and decompression implementation
// Path: Falcon/src/gpu/Falcon_float_pipeline.cu

#include "Falcon_float_pipeline.cuh"
#include <algorithm>
#include <iostream>
#include <cstring>

// Destructor
FalconPipeline_32::~FalconPipeline_32() {
}

// Helper: create a CompressedData structure from a CompressionResult
CompressedData_32 FalconPipeline_32::createCompressedData(
    const CompressionResult_32& compResult,
    const ProcessedData_32& originalData) {
    
    CompressedData_32 compData;
    compData.cmpBytes = originalData.cmpBytes;
    compData.totalCompressedSize = compResult.analysis.total_compressed_size;
    compData.totalElements = originalData.nbEle;
    compData.chunkSizes = compResult.chunkSizes;
    compData.chunkElementCounts = compResult.chunkElementCounts;
    compData.totalChunks = compResult.totalChunks;
    
    return compData;
}

// Cleanup helper for host-side resources
void cleanup_data_32(ProcessedData_32 &data) {
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

// Run compression pipeline using the member NUM_STREAMS
CompressionResult_32 FalconPipeline_32::executeCompressionPipeline(
    ProcessedData_32 &data,
    size_t chunkSize) {
    
    return executeCompressionPipelineImpl(data, chunkSize, NUM_STREAMS);
}

// Run compression pipeline with an explicit number of streams
CompressionResult_32 FalconPipeline_32::executeCompressionPipeline(
    ProcessedData_32 &data,
    size_t chunkSize,
    int numStreams) {
    
    return executeCompressionPipelineImpl(data, chunkSize, numStreams);
}

// Internal implementation of the compression pipeline
CompressionResult_32 FalconPipeline_32::executeCompressionPipelineImpl(
    ProcessedData_32 &data,
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
    cudaCheckError_32(cudaHostAlloc((void**)&locCmpSize, 
        sizeof(unsigned int) * (totalChunks + 2), cudaHostAllocDefault));
    
    // Set guard values
    locCmpSize[0] = 0xDEADBEEF;
    locCmpSize[totalChunks + 1] = 0xCAFEBABE;
    
    unsigned int *h_cmp_offset;
    cudaCheckError_32(cudaHostAlloc((void**)&h_cmp_offset, 
        sizeof(unsigned int) * (totalChunks + 1), cudaHostAllocDefault));
    h_cmp_offset[0] = 0;
    
    bool *of_rd = new bool[totalChunks + numStreams]();
    of_rd[0] = true;
    
    size_t *chunkElementCounts;
    cudaCheckError_32(cudaHostAlloc((void**)&chunkElementCounts, 
        sizeof(size_t) * totalChunks, cudaHostAllocDefault));
    
    std::vector<size_t> chunkSizes(totalChunks);
    std::vector<size_t> chunkElementCountsVec(totalChunks);
    
    // Create stream pool and events
    const int MAX_EVENTS_PER_TYPE = totalChunks + numStreams;
    
    cudaStream_t *streams = new cudaStream_t[numStreams];
    cudaEvent_t *evSize = new cudaEvent_t[MAX_EVENTS_PER_TYPE];
    cudaEvent_t *evData = new cudaEvent_t[MAX_EVENTS_PER_TYPE];
    Stage_32 *stage = new Stage_32[numStreams];
    
    for (int i = 0; i < numStreams; ++i) {
        cudaCheckError_32(cudaStreamCreate(&streams[i]));
        stage[i] = IDLE_32;
    }
    
    for (int i = 0; i < MAX_EVENTS_PER_TYPE; ++i) {
        cudaCheckError_32(cudaEventCreateWithFlags(&evSize[i], cudaEventDisableTiming));
        cudaCheckError_32(cudaEventCreateWithFlags(&evData[i], cudaEventDisableTiming));
    }
    
    // Allocate fixed device buffers for each stream
    float **d_in = new float*[numStreams];
    unsigned char **d_out = new unsigned char*[numStreams];
    
    for (int i = 0; i < numStreams; ++i) {
        cudaCheckError_32(cudaMalloc(&d_in[i], chunkSize * sizeof(float)));
        cudaCheckError_32(cudaMalloc(&d_out[i], chunkSize * sizeof(float)));
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
                case IDLE_32:
                    if (processedEle < data.nbEle) {
                        size_t todo = std::min(chunkSize, data.nbEle - processedEle);
                        if(todo == 0) continue;
                        
                        progress = 1;
                        chunkIDX[s] = completedChunks;
                        completedChunks++;
                        
                        chunkElementCounts[chunkIDX[s]] = todo;
                        
                        // Asynchronous host-to-device copy
                        cudaCheckError_32(cudaMemcpyAsync(
                            d_in[s],
                            data.oriData + processedEle,
                            todo * sizeof(float),
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
                        stage[s] = SIZE_PENDING_32;
                        cudaCheckError_32(cudaEventRecord(evSize[chunkIDX[s]], streams[s]));
                    }
                    break;
                    
                case SIZE_PENDING_32:
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
                        cudaCheckError_32(cudaMemcpyAsync(
                            data.cmpBytes + h_cmp_offset[idx],
                            d_out[s],
                            compressedBytes,
                            cudaMemcpyDeviceToHost,
                            streams[s]));
                        
                        h_cmp_offset[idx + 1] = h_cmp_offset[idx] + compressedBytes;
                        of_rd[idx + 1] = true;
                        chunkSizes[idx] = compressedBytes;
                        chunkElementCountsVec[idx] = chunkElementCounts[idx];
                        
                        cudaCheckError_32(cudaEventRecord(evData[chunkIDX[s]], streams[s]));
                        stage[s] = DATA_PENDING_32;
                    }
                    break;
                    
                case DATA_PENDING_32:
                    if (cudaEventQuery(evData[chunkIDX[s]]) == cudaSuccess) {
                        unsigned int compressedBytes = (locCmpSize[chunkIDX[s] + 1] + 7) / 8;
                        totalCmpSize += compressedBytes;
                        progress = 1;
                        stage[s] = IDLE_32;
                        active -= 1;
                    }
                    break;
            }
        }
        
        // Avoid busy waiting
        if (!progress) {
            for (int i = 0; i < numStreams; i++) {
                cudaCheckError_32(cudaStreamSynchronize(streams[i]));
            }
        }
    }
    
    // Wait for all streams to finish
    for (int i = 0; i < std::min(numStreams, (int)totalChunks); i++) {
        cudaCheckError_32(cudaStreamSynchronize(streams[i]));
        cudaCheckError_32(cudaEventSynchronize(evData[i]));
    }
    
    cudaEventRecord(global_end_event);
    cudaEventSynchronize(global_end_event);
    
    // Compute total time
    float totalTime;
    cudaEventElapsedTime(&totalTime, global_start_event, global_end_event);
    
    // Compute compression ratio
    double compressionRatio = totalCmpSize / static_cast<double>(data.nbEle * sizeof(float));
    
    // Create analysis result
    PipelineAnalysis_32 analysis;
    analysis.compression_ratio = compressionRatio;
    analysis.total_compressed_size = totalCmpSize;
    analysis.total_size = data.nbEle * sizeof(float) / 1024.0 / 1024.0;
    analysis.comp_time = totalTime;
    analysis.comp_throughout = (data.nbEle * sizeof(float) / 1024.0 / 1024.0 / 1024.0) / 
                              (totalTime / 1000.0);
    analysis.chunk_size = chunkSize;
    *data.cmpSize = totalCmpSize;
    
    // Free device memory
    for (int i = 0; i < numStreams; i++) {
        cudaCheckError_32(cudaFree(d_out[i]));
        cudaCheckError_32(cudaFree(d_in[i]));
    }
    
    // Destroy streams and events
    for (int i = 0; i < numStreams; i++) {
        cudaCheckError_32(cudaStreamDestroy(streams[i]));
    }
    for (int i = 0; i < MAX_EVENTS_PER_TYPE; i++) {
        cudaCheckError_32(cudaEventDestroy(evSize[i]));
        cudaCheckError_32(cudaEventDestroy(evData[i]));
    }
    
    // Free host memory
    cudaCheckError_32(cudaFreeHost(locCmpSize));
    cudaCheckError_32(cudaFreeHost(h_cmp_offset));
    cudaCheckError_32(cudaFreeHost(chunkElementCounts));
    delete[] of_rd;
    
    // Destroy global events
    cudaCheckError_32(cudaEventDestroy(global_start_event));
    cudaCheckError_32(cudaEventDestroy(global_end_event));
    cudaCheckError_32(cudaEventDestroy(init_start_event));
    cudaCheckError_32(cudaEventDestroy(init_end_event));
    
    // Free dynamically allocated arrays
    delete[] streams;
    delete[] evSize;
    delete[] evData;
    delete[] stage;
    delete[] d_in;
    delete[] d_out;
    
    // Create and return the final result
    CompressionResult_32 result;
    result.analysis = analysis;
    result.chunkSizes = std::move(chunkSizes);
    result.chunkElementCounts = std::move(chunkElementCountsVec);
    result.totalChunks = completedChunks;
    
    return result;
}

// Run decompression pipeline using the member NUM_STREAMS
PipelineAnalysis_32 FalconPipeline_32::executeDecompressionPipeline(
    const CompressionResult_32& compResult,
    ProcessedData_32 &decompData,
    bool visualize) {
    
    return executeDecompressionPipelineImpl(compResult, decompData, NUM_STREAMS, visualize);
}

// Run decompression pipeline with an explicit number of streams
PipelineAnalysis_32 FalconPipeline_32::executeDecompressionPipeline(
    const CompressionResult_32& compResult,
    ProcessedData_32 &decompData,
    int numStreams,
    bool visualize) {
    
    return executeDecompressionPipelineImpl(compResult, decompData, numStreams, visualize);
}

// Internal implementation of the decompression pipeline
PipelineAnalysis_32 FalconPipeline_32::executeDecompressionPipelineImpl(
    const CompressionResult_32& compResult,
    ProcessedData_32 &decompData,
    int numStreams,
    bool visualize) {
    
    CompressedData_32 compData = createCompressedData(compResult, decompData);
    
    cudaDeviceSynchronize();
    
    cudaEvent_t global_start_event, global_end_event;
    cudaEventCreate(&global_start_event);
    cudaEventCreate(&global_end_event);
    
    // Host-side memory allocation
    size_t *streamChunkIds;
    cudaCheckError_32(cudaHostAlloc((void**)&streamChunkIds, 
        sizeof(size_t) * numStreams, cudaHostAllocDefault));
    
    size_t *streamOutputOffsets;
    cudaCheckError_32(cudaHostAlloc((void**)&streamOutputOffsets, 
        sizeof(size_t) * numStreams, cudaHostAllocDefault));
    
    // Create stream pool and events
    cudaStream_t *streams = new cudaStream_t[numStreams];
    cudaEvent_t *evSize = new cudaEvent_t[numStreams];
    cudaEvent_t *evData = new cudaEvent_t[numStreams];
    Stage_32 *stage = new Stage_32[numStreams];
    
    for (int i = 0; i < numStreams; ++i) {
        cudaCheckError_32(cudaStreamCreate(&streams[i]));
        cudaCheckError_32(cudaEventCreateWithFlags(&evSize[i], cudaEventDisableTiming));
        cudaCheckError_32(cudaEventCreateWithFlags(&evData[i], cudaEventDisableTiming));
        stage[i] = IDLE_32;
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
    float **d_out = new float*[numStreams];
    
    for (int i = 0; i < numStreams; ++i) {
        cudaCheckError_32(cudaMalloc(&d_in[i], maxCompressedSize));
        cudaCheckError_32(cudaMalloc(&d_out[i], maxChunkSize * sizeof(float)));
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
                case IDLE_32:
                    if (processedChunks < compData.totalChunks) {
                        active += 1;
                        size_t chunkId = processedChunks;
                        size_t currentChunkElements = compData.chunkElementCounts[chunkId];
                        size_t currentChunkCompressedSize = compData.chunkSizes[chunkId];
                        
                        streamChunkIds[s] = chunkId;
                        streamOutputOffsets[s] = processedElements;
                        
                        size_t compressedDataOffset = compressedDataOffsets[chunkId];
                        
                        // Asynchronous host-to-device copy of compressed data
                        cudaCheckError_32(cudaMemcpyAsync(
                            d_in[s],
                            compData.cmpBytes + compressedDataOffset,
                            currentChunkCompressedSize,
                            cudaMemcpyHostToDevice,
                            streams[s]));
                        
                        // Launch decompression kernel
                        FalconDecompressor Falcon;
                        Falcon.Falcon_decompress_stream_optimized(
                            d_out[s],
                            d_in[s],
                            currentChunkElements,
                            currentChunkCompressedSize,
                            streams[s]);
                        
                        cudaCheckError_32(cudaEventRecord(evSize[s], streams[s]));
                        
                        processedChunks++;
                        processedElements += currentChunkElements;
                        
                        size_t outputOffset = streamOutputOffsets[s];
                        
                        // Asynchronous device-to-host copy of decompressed data
                        cudaCheckError_32(cudaMemcpyAsync(
                            decompData.decData + outputOffset,
                            d_out[s],
                            currentChunkElements * sizeof(float),
                            cudaMemcpyDeviceToHost,
                            streams[s]));
                        
                        cudaCheckError_32(cudaEventRecord(evData[s], streams[s]));
                        stage[s] = DATA_PENDING_32;
                    }
                    break;
                    
                case DATA_PENDING_32:
                    if (cudaEventQuery(evData[s]) == cudaSuccess) {
                        size_t chunkId = streamChunkIds[s];
                        size_t currentChunkElements = compData.chunkElementCounts[chunkId];
                        totalDecompSize += currentChunkElements * sizeof(float);
                        
                        stage[s] = IDLE_32;
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
        cudaCheckError_32(cudaStreamSynchronize(streams[i]));
        cudaCheckError_32(cudaEventSynchronize(evData[i]));
    }
    
    cudaEventRecord(global_end_event);
    cudaEventSynchronize(global_end_event);
    
    // Compute total time
    float totalTime;
    cudaEventElapsedTime(&totalTime, global_start_event, global_end_event);
    
    // Create and return analysis result
    PipelineAnalysis_32 result;
    result.compression_ratio = totalDecompSize / static_cast<double>(compData.totalCompressedSize);
    result.total_compressed_size = compData.totalCompressedSize / 1024.0 / 1024.0;
    result.total_size = totalDecompSize / 1024.0 / 1024.0;
    result.decomp_time = totalTime;
    result.decomp_throughout = (totalDecompSize / 1024.0 / 1024.0 / 1024.0) / (totalTime / 1000.0);
    
    // Optionally verify decompressed data
    if (visualize) {
        for(int z=0,tmp=0,k=0;z<3;z++) {   
            for(size_t i=0,j=tmp;i<3;i++) {
                while(decompData.oriData[j]==decompData.decData[j]&&j<processedElements) {
                    j++;
                    if(j>=processedElements) {
                        break;
                    }
                }
                if(j>=processedElements) {
                    printf("success!!\n");   
                    break;
                }
                while(tmp<=j) {
                    tmp+=compData.chunkElementCounts[k];
                    k++;
                }
                printf("chunk:%d,idx: %zu ,ori: %.8f , dec: %.8f \n",
                    k-1, j, decompData.oriData[j], decompData.decData[j]);
                j++;
            }
        }
    }
    
    // Free device memory
    for (int i = 0; i < numStreams; i++) {
        cudaCheckError_32(cudaFree(d_out[i]));
        cudaCheckError_32(cudaFree(d_in[i]));
    }
    
    // Destroy streams and events
    for (int i = 0; i < numStreams; i++) {
        cudaCheckError_32(cudaEventDestroy(evSize[i]));
        cudaCheckError_32(cudaEventDestroy(evData[i]));
        cudaCheckError_32(cudaStreamDestroy(streams[i]));
    }
    
    // Free host memory
    cudaCheckError_32(cudaFreeHost(streamChunkIds));
    cudaCheckError_32(cudaFreeHost(streamOutputOffsets));
    
    // Destroy global events
    cudaCheckError_32(cudaEventDestroy(global_start_event));
    cudaCheckError_32(cudaEventDestroy(global_end_event));
    
    // Free dynamically allocated arrays
    delete[] streams;
    delete[] evSize;
    delete[] evData;
    delete[] stage;
    delete[] d_in;
    delete[] d_out;
    
    return result;
}