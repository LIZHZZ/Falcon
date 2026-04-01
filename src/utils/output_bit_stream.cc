#include "output_bit_stream.h"
# include<iostream>
// Initialize data_ with the given buffer size
OutputBitStream::OutputBitStream(uint32_t buffer_size) {
    data_ = Array<uint32_t>(buffer_size / 4 + 1);
    buffer_ = 0;
    cursor_ = 0;
    bit_in_buffer_ = 0;
}

// Write bits: content + length
uint32_t OutputBitStream::Write(uint64_t content, uint32_t len) {
    if (len > 64) {
        std::cerr << "Error: Attempt to write more than 64 bits." << std::endl;
        return 0; // Prevent overflow
    }
    
    // Grow buffer_ if there is not enough space
    if (cursor_ >= data_.length()) {
        int newsize=(cursor_ + (len + 7) / 8);
        if (newsize > data_.length()) 
        {
            Array<uint32_t> newData(newsize);
            std::copy(data_.begin(), data_.end(), newData.begin());
            data_ = std::move(newData);
        }
        std::cerr << "Error: cursor exceeds data array size." << std::endl;
    }

    content <<= (64 - len);                  // Shift left so valid bits align to the most significant bits
    buffer_ |= (content >> bit_in_buffer_);  // Write into buffer
    bit_in_buffer_ += len;                   // Update bit count in buffer

    // Check whether bit_in_buffer_ exceeds 32
    if (bit_in_buffer_ >= 32) {             
        data_[cursor_++] = (buffer_ >> 32); // Store high 32 bits of buffer into data_
        buffer_ <<= 32;                     // Shift buffer left
        bit_in_buffer_ -= 32;               // Update remaining bits
    }
    return len;
}

// uint32_t OutputBitStream::Write(uint64_t content, uint32_t len) {
//     content <<= (64 - len);                 // Shift left so valid bits align to the most significant bits
//     buffer_ |= (content >> bit_in_buffer_); // Write into buffer
//     bit_in_buffer_ += len;                  // Update bit count in buffer
//     if (bit_in_buffer_ >= 32) {             // When buffer is full, flush to data_
//         data_[cursor_++] = (buffer_ >> 32); // Store high 32 bits of buffer into data_
//         buffer_ <<= 32;                     // Shift buffer left
//         bit_in_buffer_ -= 32;               // Update remaining bits
//     }
//     return len;
// }

// Write a long value: content + length
uint32_t OutputBitStream::WriteLong(uint64_t content, uint64_t len) {
    if (len == 0) return 0;
    if (len > 32) { // For length > 32, write in two parts
        Write(content >> (len - 32), 32);
        Write(content, len - 32);
        return len;
    }
    return Write(content, len);
}

// Write an int value: content + length
uint32_t OutputBitStream::WriteInt(uint32_t content, uint32_t len) {
    return Write(static_cast<uint64_t>(content), len);
}

// Write a single bit
uint32_t OutputBitStream::WriteBit(bool bit) {
    return Write(static_cast<uint64_t>(bit), 1);
}

// Write a single byte
uint32_t OutputBitStream::WriteByte(uint8_t bit) {
    return Write(static_cast<uint64_t>(bit), 8);
}

// Get a buffer of the specified length (in bytes) from data_ and return it
Array<uint8_t> OutputBitStream::GetBuffer(uint32_t len) {
    Array<uint8_t> ret(len);
    for (auto &blk : data_) blk = htobe32(blk); // Convert each 32-bit block to big-endian order
    __builtin_memcpy(ret.begin(), data_.begin(), len); // Copy data into result buffer
    return ret;
}

// Store remaining bits in buffer_ and clear it
void OutputBitStream::Flush() {
    if (bit_in_buffer_) {
        data_[cursor_++] = buffer_ >> 32;
        buffer_ = 0;
        bit_in_buffer_ = 0;
    }
}
// Reset internal state
void OutputBitStream::Refresh() {
    cursor_ = 0;
    bit_in_buffer_ = 0;
    buffer_ = 0;
}
