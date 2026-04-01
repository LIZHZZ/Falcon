#ifndef SERF_OUTPUT_BIT_STREAM_H
#define SERF_OUTPUT_BIT_STREAM_H

#include <cstdint>

#include "array.h"

// Output bit stream
//  Big-endian byte order
//  data_: stores encoded data blocks
//  cursor_: cursor of the current write position in data_
//  bit_in_buffer_: number of bits currently stored in the buffer_
//  buffer_: temporary buffer used for bit-wise writes
class OutputBitStream {
 public:
    explicit OutputBitStream(uint32_t buffer_size);

    uint32_t Write(uint64_t content, uint32_t len);

    uint32_t WriteLong(uint64_t content, uint64_t len);

    uint32_t WriteInt(uint32_t content, uint32_t len);

    uint32_t WriteBit(bool bit);

    uint32_t WriteByte(uint8_t bit);

    void Flush();

    Array<uint8_t> GetBuffer(uint32_t len);

    void Refresh();

   uint32_t GetBufferSize() const {
    return cursor_;
   }
 private:
    Array<uint32_t> data_;    // Encoded data blocks
    uint32_t cursor_;         // Cursor of current write position in data_
    uint32_t bit_in_buffer_;  // Number of bits currently stored in buffer_
    uint64_t buffer_;         // Temporary buffer used for bit-wise writes
};

#endif  // SERF_OUTPUT_BIT_STREAM_H
