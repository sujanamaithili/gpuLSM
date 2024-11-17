#ifndef GPU_MERGE_SORT_HELPER_H
#define GPU_MERGE_SORT_HELPER_H

#include "lsm.cuh"

template <typename Key, typename Value>
void mergeSortGPU(Pair<Key, Value>* d_arr, long int n);

template <typename Key, typename Value>
void sortBySegment(Pair<Key, Value>* d_result, int* d_maxoffset, int* d_result_offset, int numQueries);

#endif 
