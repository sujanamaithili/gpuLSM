#include "lsm.cuh"
#include "query.cuh"
#include "merge.cuh"
#include "initialize.cuh"
#include "bitonicSort.cuh"
#include "reduceSum.cuh"
#include "bounds.cuh"
#include "collectElements.cuh"
#include "compact.cuh"
#include "count.cuh"
#include "exclusiveSum.cuh"
#include <cstdio>
#include <cuda.h>

template class lsmTree<int, int>;

template <typename Key, typename Value>
__host__ lsmTree<Key, Value>::lsmTree(int numLevels, int bufferSize) {
    this->numLevels = numLevels;
    this->bufferSize = bufferSize;
    this->maxSize = 0;
    this->numBatches = 0;
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
    initializeMemory<<<(maxSize + 255) / 256, 256>>>(memory, maxSize);
    cudaDeviceSynchronize();
}

template <typename Key, typename Value>
__host__ lsmTree<Key, Value>::~lsmTree() {
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

    
    bitonicSortGPU(d_buffer, batch_size);
    
    int offset = 0;
    int level_size = batch_size; //b
    int current_level = 0;
    int merged_size = batch_size;
    
    Pair<Key, Value>* m = getMemory();

    while(getNumBatches() & (1 << current_level)){
        Pair<Key, Value>* cur = getMemory() + offset;

        merged_size += level_size;

        d_buffer = merge(d_buffer, level_size, m + offset, level_size);
        cudaMemset(cur, 0, level_size * sizeof(Pair<Key, Value>));

        offset += level_size;
        current_level++;
        level_size <<= 1;
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
    int num_levels = getNumLevels();
    int buffer_size = getBufferSize();
    int num_batches = getNumBatches();

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    queryKeysKernel<<<blocksPerGrid, threadsPerBlock>>>(d_keys, d_results, d_foundFlags, size, d_memory, num_levels, buffer_size, num_batches);

    // Copy results back to host
    cudaMemcpy(results, d_results, size * sizeof(Value), cudaMemcpyDeviceToHost);
    cudaMemcpy(foundFlags, d_foundFlags, size * sizeof(bool), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_keys);
    cudaFree(d_results);
    cudaFree(d_foundFlags);
}

template <typename Key, typename Value>
__host__ bool lsmTree<Key, Value>::deleteKeys(const Key* keys, int batch_size)
{

    Pair<Key, Value>* h_buffer = new Pair<Key, Value>[batch_size];
    for (int i = 0; i < batch_size; ++i) {
        h_buffer[i] = Pair<Key, Value>(keys[i], std::nullopt);
    }
    Pair<Key, Value>* d_buffer;
    cudaMalloc(&d_buffer, batch_size * sizeof(Pair<Key, Value>));
    cudaMemcpy(d_buffer, h_buffer, batch_size * sizeof(Pair<Key, Value>), cudaMemcpyHostToDevice);

    bitonicSortGPU(d_buffer, batch_size);

    int offset = 0;
    int level_size = batch_size;
    int current_level = 0;
    int merged_size = batch_size;

    Pair<Key, Value>* m = getMemory();

    while (getNumBatches() & (1 << current_level)) {
        Pair<Key, Value>* cur = getMemory() + offset;

        merged_size += level_size;

        d_buffer = merge(d_buffer, level_size, m + offset, level_size);
        cudaMemset(cur, 0, level_size * sizeof(Pair<Key, Value>));

        offset += level_size;
        current_level++;
        level_size <<= 1;
    }

    cudaMemcpy(m + offset, d_buffer, merged_size * sizeof(Pair<Key, Value>), cudaMemcpyDeviceToDevice);
    incrementBatchCounter();
    delete[] h_buffer;
    cudaFree(d_buffer);
    return true;
}

template <typename Key, typename Value>
__host__ void lsmTree<Key, Value>::countKeys(const Key* k1, const Key* k2, int numQueries, int* counts) {
    // lower and upper bounds index in every level of each query
    int* d_l;                
    int* d_u; 
    int* d_init_count;

    int numLevels = getNumLevels();
    int bufferSize = getBufferSize();
    Pair<Key, Value>* m = getMemory();

    cudaMalloc(&d_l, numQueries * numLevels * sizeof(int));
    cudaMalloc(&d_u, numQueries * numLevels * sizeof(int));
    cudaMalloc(&d_init_count, numQueries * numLevels * sizeof(int));

    // Launch kernel to find lower and upper bounds for each query on each level

    findBounds<<<numQueries, numLevels>>>(d_l, d_u, k1, k2, d_init_count, bufferSize, m, numLevels);

    int* d_offset;
    cudaMalloc(&d_offset, numQueries * numLevels * sizeof(int));
    int* d_maxoffset;
    cudaMalloc(&d_maxoffset, numQueries * sizeof(int));

    int threadsPerBlock = 256;
    int blocks = (numQueries + threadsPerBlock - 1) / threadsPerBlock;
    exclusiveSum<<<blocks, threadsPerBlock>>>(d_init_count, d_offset, d_maxoffset, numQueries, numLevels);

    
    int* d_maxResultSize;
    int reductionThreads = 256;
    int reductionBlocks = (numQueries + reductionThreads - 1) / reductionThreads;
    reduceSum<<<reductionBlocks, reductionThreads>>>(d_maxoffset, d_maxResultSize, numQueries);

    int maxResultSize;
    cudaMemcpy(&maxResultSize, d_maxResultSize, sizeof(int), cudaMemcpyDeviceToHost);

    Pair<Key, Value>* d_result;
    cudaMalloc(&d_result, maxResultSize * sizeof(Pair<Key, Value>));
    collectElements<<<numQueries, numLevels>>>(d_l, d_u, d_offset, d_result, bufferSize, m, numLevels);

    int* d_result_offset;
    cudaMalloc(&d_result_offset, numQueries * sizeof(int));
    sortBySegment(d_result, d_maxoffset, d_result_offset, numQueries);

    int* d_counts;
    cudaMalloc(&d_counts, numQueries * sizeof(int));
    count<<<blocks, threadsPerBlock>>>(d_result, d_maxoffset, d_result_offset, d_counts, numQueries);
    cudaMemcpy(counts, d_counts, numQueries * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_l);
    cudaFree(d_u);
    cudaFree(d_init_count);
    cudaFree(d_offset);
    cudaFree(d_maxoffset);
    cudaFree(d_result_offset);
    cudaFree(d_result);
    cudaFree(d_counts);
}

template <typename Key, typename Value>
__host__ void lsmTree<Key, Value>::rangeKeys(const Key* k1, const Key* k2, int numQueries, Pair<Key, Value>* range, int* counts, int* range_offset) {
    // lower and upper bounds index in every level of each query
    int* d_l;                
    int* d_u; 
    int* d_init_count;
    int numLevels = getNumLevels();
    cudaMalloc(&d_l, numQueries * numLevels * sizeof(int));
    cudaMalloc(&d_u, numQueries * numLevels * sizeof(int));
    cudaMalloc(&d_init_count, numQueries * numLevels * sizeof(int));
    // Launch kernel to find lower and upper bounds for each query on each level
    findBounds<<<numQueries, numLevels>>>(d_l, d_u, k1, k2, d_init_count);
    int* d_offset;
    cudaMalloc(&d_offset, numQueries * numLevels * sizeof(int));
    int* d_maxoffset;
    cudaMalloc(&d_maxoffset, numQueries * sizeof(int));
    int threadsPerBlock = 256;
    int blocks = (numQueries + threadsPerBlock - 1) / threadsPerBlock;
    exclusiveSum<<<blocks, threadsPerBlock>>>(d_init_count, d_offset, d_maxoffset, numQueries);
    
    int* d_maxResultSize;
    int reductionThreads = 256;
    int reductionBlocks = (numQueries + reductionThreads - 1) / reductionThreads;
    reduceSum<<<reductionBlocks, reductionThreads>>>(d_maxoffset, d_maxResultSize, numQueries);
    int maxResultSize;
    cudaMemcpy(&maxResultSize, d_maxResultSize, sizeof(int), cudaMemcpyDeviceToHost);
    Pair<Key, Value>* d_result;
    cudaMalloc(&d_result, maxResultSize * sizeof(Pair<Key, Value>));
    collectElements<<<numQueries, numLevels>>>(d_l, d_u, d_offset, d_result);
    int* d_result_offset;
    cudaMalloc(&d_result_offset, numQueries * sizeof(int));
    sortBySegment(d_result, d_maxoffset, d_result_offset, numQueries);
    int* d_counts;
    cudaMalloc(&d_counts, numQueries * sizeof(int));
    Pair<Key, Value>* d_range;
    cudaMalloc(&d_range, maxResultSize * sizeof(Pair<Key, Value>));
    compact<<<blocks, threadsPerBlock>>>(d_result, d_maxoffset, d_result_offset, d_range, d_counts, numQueries);
    
    cudaMemcpy(range, d_range, maxResultSize * sizeof(Pair<Key, Value>), cudaMemcpyDeviceToHost);
    cudaMemcpy(counts, d_counts, numQueries * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(range_offset, d_result_offset, numQueries * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_l);
    cudaFree(d_u);
    cudaFree(d_init_count);
    cudaFree(d_offset);
    cudaFree(d_maxoffset);
    cudaFree(d_result_offset);
    cudaFree(d_result);
    cudaFree(d_counts);
    cudaFree(d_range);
}

template <typename Key, typename Value>
__host__ void lsmTree<Key, Value>::printLevel(int level) const {
    if (level >= numLevels) {
        printf("Error: Level %d does not exist. Tree has %d levels.\n", level, numLevels);
        return;
    }

    // Calculate the offset and size for the specified level
    int offset = 0;
    int level_size = bufferSize;
    for (int i = 0; i < level; i++) {
        offset += level_size;
        level_size <<= 1;  // Double the size for each level
    }

    // Create host memory to copy the level data
    Pair<Key, Value>* h_level = new Pair<Key, Value>[level_size];
    cudaError_t status = cudaMemcpy(h_level, memory + offset,
                                level_size * sizeof(Pair<Key, Value>),
                                cudaMemcpyDeviceToHost);

    if (status != cudaSuccess) {
        printf("Error copying level data from device: %s\n", cudaGetErrorString(status));
        delete[] h_level;
        return;
    }

    printf("\nLevel %d (size: %d):\n", level, level_size);
    printf("----------------------------------------\n");

    // Count valid entries (entries with non-empty keys)
    int numEntries = 0;
    for (int i = 0; i < level_size; i++) {
        if (h_level[i].first.has_value()) {
            numEntries++;
        }
    }

    if (numEntries == 0) {
        printf("Empty level\n");
    } else {
        printf("Index\tKey\tValue\n");
        for (int i = 0; i < level_size; i++) {
            if (!h_level[i].isKeyEmpty()) {
                printf("%d\t%d\t", i, *(h_level[i].first));
                if (!h_level[i].isValueTombstone()) {
                    printf("%d\n", *(h_level[i].second));
                } else {
                    printf("tombstone\n");
                }
            }
        }
    }

    printf("Total entries: %d/%d\n", numEntries, level_size);
    printf("----------------------------------------\n");

    delete[] h_level;
}

template <typename Key, typename Value>
__host__ void lsmTree<Key, Value>::printAllLevels() const {
    printf("\nLSM Tree Structure:\n");
    printf("==================\n");
    printf("Number of levels: %d\n", numLevels);
    printf("Buffer size: %d\n", bufferSize);
    printf("Number of batches processed: %d\n", numBatches);
    printf("==================\n");

    for (int i = 0; i < numLevels; i++) {
        printLevel(i);
    }
}

