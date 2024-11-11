#include "gpu_lsm_tree.h"
#include <cstdio>
#include <cuda_runtime.h>

template <typename Key, typename Value>
__host__ __device__ lsmTree<Key, Value>::lsmTree(int numLevels, int bufferSize) {
    this->numLevels = numLevels;
    this->bufferSize = bufferSize;
    this->maxSize = 0;

    // Calculate maxSize based on the number of levels and buffer size.
    for (int i = 0; i < numLevels; ++i) {
        this->maxSize += (bufferSize << i);
    }

    // Allocate memory on the GPU.
    cudaError_t status = cudaMalloc(&memory, maxSize * sizeof(Pair<Key, Value>));
    if (status != cudaSuccess) {
        printf("Error allocating memory for LSM tree: %s\n", cudaGetErrorString(status));
        #ifndef __CUDA_ARCH__
        // Only call exit if on the host
        exit(1);
        #endif
    }
}

template <typename Key, typename Value>
__host__ __device__ lsmTree<Key, Value>::~lsmTree() {
    if (memory != nullptr) {
        cudaFree(memory);
    }
}


template <typename Key, typename Value>
__host__ Value* lsm<Key, Value>::queryBatch(Key* batch, int batch_size)
{
    Value* results = new Value[batch_size];
    
    
}

template <typename Key, typename Value>
__host__ bool lsm<Key, Value>::updateKeys(Pair<key, Valye>* kv, int batch_size)
{
    
    Pair<Key, Value>* d_buffer;
    cudaMalloc(&d_buffer, batch_size * sizeof(Pair<Key, Value>));
    cudaMemcpy(d_buffer, kv, batch_size * sizeof(Pair<Key, Value>), cudaMemcpyHostToDevice);
    cudaMalloc(&tempd_buffer, batch_size * sizeof(Pair<Key, Value>));
    cub::DeviceRadixSort::SortPairs(tempd_buffer, batch_size, d_buffer, d_buffer, batch_size);

    int offset = 0;
    int level_size = batch_size; //b
    int current_level = 0;
    int merged_size = batch_size;
    
    Pair<Key, Value>* m = getMemory();

    while(getNumBatches() & (1 << currentLevel)){
        Pair<Key, Value>* cur = getMemory() + offset;

        merged_size += level_size;

        d_buffer = mgpu::merge(m + offset, level_size, d_buffer, level_size);
        cudaMemset(cur, 0, level_size * sizeof(Pair<Key, Value>));

        offset += level_size;
        current_level++;
        level_size <<= 1
    }
    
    cudaMemcpy(m + offset, d_buffer, merged_size * sizeof(Pair<Key, Value>), cudaMemcpyDeviceToDevice);
    incrementBatchCounter();
    cudaFree(d_buffer);
    return true;
}

