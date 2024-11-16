#ifndef GPU_REDUCE_SUM_HELPER_H
#define GPU_REDUCE_SUM_HELPER_H

#include "lsm.cuh"

template <typename Key, typename Value>
void reduceSum(const int* input, int* result, int size);

#endif 