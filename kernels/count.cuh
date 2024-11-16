#ifndef GPU_COUNT_HELPER_H
#define GPU_COUNT_HELPER_H

#include "lsm.cuh"

template <typename Key, typename Value>
void count(const Pair<Key, Value>* d_result, const int* d_maxoffset, const int* d_result_offset, int* d_counts, int numQueries);

#endif