//
// cuCompressor/include/GDFDeCompressor.cuh
//
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>

#include <thread> 
#pragma once
#include <vector>
#include <cstdint>

// Decompression class
class FalconDecompressor {
public:
    void Falcon_decompress(double* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, cudaStream_t stream);
    void Falcon_decompress_stream_optimized(
            double* d_decData,          // Decompression output buffer
            unsigned char* d_cmpBytes,  // Compressed input buffer (device side)
            size_t nbEle,               // Number of original elements
            size_t cmpSize,             // Size of compressed data
            cudaStream_t stream) ;
    // void Falcon_decompress_ultra_optimized(
    //         double* d_decData,
    //         unsigned char* d_cmpBytes,
    //         size_t nbEle,
    //         size_t cmpSize,
    //         cudaStream_t stream);
    void decompress(const std::vector<unsigned char>& compressedData, std::vector<double>& output,int numDatas);
    void Falcon_decompress_no_pack(
        double* d_decData,          // Decompression output buffer
        unsigned char* d_cmpBytes,  // Compressed input buffer (device side)
        size_t nbEle,               // Number of original elements
        size_t cmpSize,             // Size of compressed data
        cudaStream_t stream         // CUDA stream
    );
};


// Host-side helper class for reading compressed data bit by bit
class BitReader {
public:
    BitReader(const std::vector<unsigned char>& buffer) : buffer(buffer), bitPos(0) {}

    // Read n bits and return them as a uint64_t
    uint64_t readBits(int n) {
        uint64_t value = 0;
        for(int i = 0; i < n; ++i) {
            size_t byteIdx = bitPos / 8;
            size_t bitIdx = bitPos % 8;
            if(byteIdx >= buffer.size()) break;
            uint8_t bit = (buffer[byteIdx] >> bitIdx) & 1;
            value |= (static_cast<uint64_t>(bit) << i);
            bitPos++;
        }
        return value;
    }
    uint64_t readBits(int begin,int n) {
        uint64_t value = 0;
        for(int i = 0; i < n; ++i) {
            size_t byteIdx = begin / 8;
            size_t bitIdx = begin % 8;
            if(byteIdx >= buffer.size()) break;
            uint8_t bit = (buffer[byteIdx] >> bitIdx) & 1;
            value |= (static_cast<uint64_t>(bit) << i);
            begin++;
        }
        return value;
    }
    // Skip n bits
    void advance(int n) {
        bitPos += n;
    }

    // Get current bit position
    size_t getBitPos() const {
        return bitPos;
    }

private:
    const std::vector<unsigned char>& buffer;
    size_t bitPos;
};

class BitReader0 {
public:
    // Original vector constructor - kept for backward compatibility
    BitReader0(const std::vector<unsigned char>& buffer) : 
        bufferPtr(buffer.data()), bufferSize(buffer.size()), bitPos(0), ownsBuffer(false) {}
    
    // New pointer constructor - supports using raw pointers directly
    BitReader0(const unsigned char* buffer, size_t size) : 
        bufferPtr(buffer), bufferSize(size), bitPos(0), ownsBuffer(false) {}

    // Read n bits and return them as a uint64_t
    uint64_t readBits(int n) {
        uint64_t value = 0;
        for(int i = 0; i < n; ++i) {
            size_t byteIdx = bitPos / 8;
            size_t bitIdx = bitPos % 8;
            if(byteIdx >= bufferSize) break;
            uint8_t bit = (bufferPtr[byteIdx] >> bitIdx) & 1;
            value |= (static_cast<uint64_t>(bit) << i);
            bitPos++;
        }
        return value;
    }
    
    // Read n bits starting from the specified bit position
    uint64_t readBits(int begin, int n) {
        uint64_t value = 0;
        for(int i = 0; i < n; ++i) {
            size_t byteIdx = begin / 8;
            size_t bitIdx = begin % 8;
            if(byteIdx >= bufferSize) break;
            uint8_t bit = (bufferPtr[byteIdx] >> bitIdx) & 1;
            value |= (static_cast<uint64_t>(bit) << i);
            begin++;
        }
        return value;
    }
    
    // Skip n bits
    void advance(int n) {
        bitPos += n;
    }

    // Get current bit position
    size_t getBitPos() const {
        return bitPos;
    }
    
    // Get buffer size
    size_t getBufferSize() const {
        return bufferSize;
    }

private:
    const unsigned char* bufferPtr;  // Unified pointer-based buffer
    size_t bufferSize;               // Buffer size
    size_t bitPos;                   // Current bit position
    bool ownsBuffer;                 // Whether this object owns the buffer (for potential future memory management)
};

