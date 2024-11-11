#ifndef GPU_MERGE_HELPER_H
#define GPU_MERGE_HELPER_H

#include "src/lsm.cuh"

template <typename Key, typename Value>
Pair<Key, Value>* merge(Pair<Key, Value>* d_arr1, int size1, Pair<Key, Value>* d_arr2, int size2);

#endif 