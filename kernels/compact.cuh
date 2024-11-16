#ifndef GPU_COMPACT_HELPER_H
#define GPU_COMPACT_HELPER_H

#include "lsm.cuh"

template <typename Key, typename Value>
void compact(const Pair<Key, Value>* d_result, const int* d_maxoffset, const int* d_result_offset, Pair<Key, Value>* d_range, int* d_counts, int numQueries);

#endif