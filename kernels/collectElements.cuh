#ifndef GPU_COLLECT_ELEMENTS_HELPER_H
#define GPU_COLLECT_ELEMENTS_HELPER_H

#include "lsm.cuh"

template <typename Key, typename Value>
void collectElements(const int* d_l, const int* d_u, const int* d_offset, Pair<Key, Value>* d_result);

#endif