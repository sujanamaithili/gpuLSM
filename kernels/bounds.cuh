#ifndef GPU_BOUNDS_HELPER_H
#define GPU_BOUNDS_HELPER_H

#include "lsm.cuh"

template <typename Key>
__global__ void findBounds(int* d_l, int* d_u, const Key* k1, const Key* k2, int* d_init_count, int bufferSize);

template <typename Key, typename Value>
int lowerBound(int level, Key key, int bufferSize);

template <typename Key, typename Value>
int upperBound(int level, Key key, int bufferSize);
 
#endif 
