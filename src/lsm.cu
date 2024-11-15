#include "lsm.cuh"
#include "query.cuh"
#include "merge.cuh"
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
__host__ bool lsmTree<Key, Value>::updateKeys(const Pair<Key, Value>* kv, int batch_size)
{
    
    Pair<Key, Value>* d_buffer;
    cudaMalloc(&d_buffer, batch_size * sizeof(Pair<Key, Value>));
    cudaMemcpy(d_buffer, kv, batch_size * sizeof(Pair<Key, Value>), cudaMemcpyHostToDevice);

    
    mergeSortGPU(d_buffer, batch_size)
    
    int offset = 0;
    int level_size = batch_size; //b
    int current_level = 0;
    int merged_size = batch_size;
    
    Pair<Key, Value>* m = getMemory();

    while(getNumBatches() & (1 << currentLevel)){
        Pair<Key, Value>* cur = getMemory() + offset;

        merged_size += level_size;

        d_buffer = merge(m + offset, level_size, d_buffer, level_size);
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


template <typename Key, typename Value>
__host__ void lsmTree<Key, Value>::queryKeys(const Key* keys, int size, Value* results, bool* foundFlags) {
    // Allocate device memory for keys, results, and found flags
    Key* d_keys;
    Value* d_results;
    bool* d_foundFlags;

    cudaMalloc(&d_keys, size * sizeof(Key));
    cudaMalloc(&d_results, size * sizeof(Value));
    cudaMalloc(&d_foundFlags, size * sizeof(bool));
    cudaMemcpy(d_keys, keys, size * sizeof(Key), cudaMemcpyHostToDevice);

    // Get device pointer to the LSM tree memory
    Pair<Key, Value>* d_memory = getMemory();
    int num_levels = numLevels;
    int buffer_size = bufferSize;

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    queryKeysKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_keys, d_results, d_foundFlags, size, d_memory, num_levels, buffer_size);

    // Copy results back to host
    cudaMemcpy(results, d_results, size * sizeof(Value), cudaMemcpyDeviceToHost);
    cudaMemcpy(foundFlags, d_foundFlags, size * sizeof(bool), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_keys);
    cudaFree(d_results);
    cudaFree(d_foundFlags);
}

template <typename Key, typename Value>
__host__ void lsmTree<Key, Value>::countKeys(const Key* k1, const Key* k2, int numQueries, int* counts) {
    // lower and upper bounds index in every level of each query
    int* d_l;                
    int* d_u; 
    int* d_init_count;

    cudaMalloc(&d_l, numQueries * numLevels * sizeof(int));
    cudaMalloc(&d_u, numQueries * numLevels * sizeof(int));
    cudaMalloc(&d_init_count, numQueries * numLevels * sizeof(int));

    // Launch kernel to find lower and upper bounds for each query on each level
    findBounds<<<numQueries, numLevels>>>(d_l, d_u, k1, k2, d_init_count);

    int* d_offset;
    cudaMalloc(&d_offset, numQueries * numLevels * sizeof(int));

    int threadsPerBlock = 256;
    int blocks = (numQueries + threadsPerBlock - 1) / threadsPerBlock;
    exclusiveSum<<<blocks, threadsPerBlock>>>(d_init_count, d_offset);

    cudaFree(d_l);
    cudaFree(d_u);
    cudaFree(d_init_count);
}
