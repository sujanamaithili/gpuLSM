#include <lsm.cuh>
#include <cuda.h>
#include <cstdio>

template <typename Key, typename Value>
__global__ void initializeMemory(Pair<Key, Value>* memory, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        memory[idx].setKeyEmpty();
        memory[idx].setValueTombstone();
    }
}

// Explicit instantiation for <int, int>
template __global__ void initializeMemory<int, int>(Pair<int, int>*, int);
