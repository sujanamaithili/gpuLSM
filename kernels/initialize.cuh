#ifndef GPU_INITIALIZE_HELPER_H
#define GPU_INITIALIZE_HELPER_H

#include "lsm.cuh"

template <typename Key, typename Value>
__global__ void initializeMemory(Pair<Key, Value>* memory, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        memory[idx].setKeyEmpty();
        memory[idx].setValueTombstone();
    }
}

#endif 
