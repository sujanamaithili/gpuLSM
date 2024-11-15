#ifndef GPU_QUERY_HELPER_H
#define GPU_QUERY_HELPER_H

#include "lsm.cuh"

template <typename Key, typename Value>
__global__ void queryKeysKernel(const Key* d_keys, Value* d_results, bool* d_foundFlags, int size, const Pair<Key, Value>* d_memory, int num_levels, int buffer_size);

#endif 
