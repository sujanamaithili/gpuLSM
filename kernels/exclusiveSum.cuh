#ifndef GPU_EXCLUSIVE_SUM_HELPER_H
#define GPU_EXCLUSIVE_SUM_HELPER_H

#include "lsm.cuh"

__global__ void exclusiveSum(const int* d_init_count, int* d_offset, int* d_maxoffset, int numQueries);

#endif