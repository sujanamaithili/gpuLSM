#ifndef GPU_REDUCE_SUM_HELPER_H
#define GPU_REDUCE_SUM_HELPER_H

#include "lsm.cuh"

__global__ reduceSum(const int* input, int* result, int size);

#endif 