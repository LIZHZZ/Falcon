//
// Created by lizhzz on 25-7-8.
//

#ifndef POST_OFFICE_SOLVER32_CUH
#define POST_OFFICE_SOLVER32_CUH
#include "BitWriter.cuh"
#include <cstdint>
#include <cstdio>

static __device__ __constant__ int kPositionLength2Bits[] = {
    0, 0, 1, 2, 2, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5,
};


static __device__ __constant__ int kPow2z[] = {1, 2, 4, 8, 16};

__device__ int initRoundAndRepresentation32(const int *distribution, //translated comment
                                          int *representation, //：representation ( 32)
                                          int *round, //：round ( 32)
                                          int *out_positions //： positions ( 32, )
);
__device__ int write_positions_device32(
    BitWriter *writer,
    const int *positions,
    int positions_len
);


#endif //POST_OFFICE_SOLVER_CUH
