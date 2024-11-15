#ifndef GPU_BITONIC_SORT_HELPER_H
#define GPU_BITONIC_SORT_HELPER_H

#include "lsm.cuh"

template <typename Key, typename Value>
void bitonicSortGPU(Pair<Key, Value>* d_arr, long int n);

#endif 
