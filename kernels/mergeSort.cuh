#ifndef GPU_MERGE_SORT_HELPER_H
#define GPU_MERGE_SORT_HELPER_H

#include "lsm.cuh"

template <typename Key, typename Value>
void mergeSortGPU(Pair<Key, Value>* d_arr, long int n);

#endif 
