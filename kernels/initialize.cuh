#ifndef GPU_INITIALIZE_HELPER_H
#define GPU_INITIALIZE_HELPER_H

#include "lsm.cuh"

template <typename Key, typename Value>
__global__ void initializeMemory(Pair<Key, Value>* memory, int size);

#endif 
