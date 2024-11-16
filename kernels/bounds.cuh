#ifndef GPU_BOUNDS_HELPER_H
#define GPU_BOUNDS_HELPER_H

#include "lsm.cuh"

template <typename Key>
void findBounds(int* d_l, int* d_u, const Key* k1, const Key* k2, int* d_init_count);

template <typename Key, typename Value>
int lowerBound(int level, Key key);

template <typename Key, typename Value>
int upperBound(int level, Key key);
 
#endif 
