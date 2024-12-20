#ifndef GPU_LSM_TREE_H
#define GPU_LSM_TREE_H

#include <cstdio>
#include <vector>
#include <cuda.h>
#include <optional>
#include <cuda_runtime.h>

template <typename T>
void printValue(const T& value) {
    std::cout << value;
}

template <typename Key, typename Value>
struct Pair {
    std::optional<Key> first;
    std::optional<Value> second;

    __host__ __device__ Pair() : first(std::nullopt), second(std::nullopt) {}
    __host__ __device__ Pair(const std::optional<Key>& a, const std::optional<Value>& b) : first(a), second(b) {}
    __host__ __device__ Pair(std::nullopt_t, std::nullopt_t) : first(std::nullopt), second(std::nullopt) {}
    __host__ __device__ Pair(const Key& a, const Value& b): first(std::optional<Key>(a)), second(std::optional<Value>(b)) {}
    __host__ __device__ void setKeyEmpty() { first = std::nullopt; }
    __host__ __device__ void setValueTombstone() { second = std::nullopt; }

    __host__ __device__ bool isKeyEmpty() const { return !first.has_value(); }
    __host__ __device__ bool isValueTombstone() const { return !second.has_value(); }
};

#include "query.cuh"
#include "merge.cuh"
#include "initialize.cuh"
#include "bitonicSort.cuh"
#include "mergeSort.cuh"
#include "reduceSum.cuh"
#include "bounds.cuh"
#include "collectElements.cuh"
#include "compact.cuh"
#include "count.cuh"
#include "exclusiveSum.cuh"

template <typename Key, typename Value>
class lsmTree {
private:
    int numLevels; // No of levels in LSM
    int bufferSize; // Size of buffer
    int maxSize; // Max Size of LSM tree
    int numBatches; // No of batches pushed each of buffer size
    bool sortStable; // true for stable sort

public:
    Pair<Key, Value>* memory; // Array of key value pairs for all levels

    __host__ lsmTree(int numLevels, int bufferSize, bool sortStable) {
        this->numLevels = numLevels;
        this->bufferSize = bufferSize;
        this->maxSize = 0;
        this->numBatches = 0;
        this->sortStable = sortStable;
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

    __host__ lsmTree(int numLevels, int bufferSize) {
        this->numLevels = numLevels;
        this->bufferSize = bufferSize;
        this->maxSize = 0;
        this->numBatches = 0;
        this->sortStable = false;
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


    __host__ ~lsmTree() {
        if (memory != nullptr) {
            cudaFree(memory);
        }
    }

    __host__ __device__ void incrementBatchCounter() { numBatches++; }
    __host__ __device__ int getNumBatches() const { return numBatches; }
    __host__ __device__ Pair<Key, Value>* getMemory() const { return memory; }
    __host__ __device__ int getMaxSize() const {return maxSize; }
    __host__ __device__ int getNumLevels() const { return numLevels; }
    __host__ __device__ int getBufferSize() const { return bufferSize; }
    __host__ __device__ int getsortStable() const { return sortStable; }

    __host__ bool updateKeys(const Pair<Key, Value>* kv, int batch_size)
    {
        
        Pair<Key, Value>* d_buffer;
        cudaMalloc(&d_buffer, batch_size * sizeof(Pair<Key, Value>));
        cudaMemcpy(d_buffer, kv, batch_size * sizeof(Pair<Key, Value>), cudaMemcpyHostToDevice);

        bool stable = getsortStable();
        if(stable){
            mergeSortGPU(d_buffer, batch_size);
        }
        else bitonicSortGPU(d_buffer, batch_size);
        
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

    __host__ bool deleteKeys(const Key* keys, int batch_size)
    {

        Pair<Key, Value>* h_buffer = new Pair<Key, Value>[batch_size];
        for (int i = 0; i < batch_size; ++i) {
            h_buffer[i] = Pair<Key, Value>(keys[i], std::nullopt);
        }
        Pair<Key, Value>* d_buffer;
        cudaMalloc(&d_buffer, batch_size * sizeof(Pair<Key, Value>));
        cudaMemcpy(d_buffer, h_buffer, batch_size * sizeof(Pair<Key, Value>), cudaMemcpyHostToDevice);

        bool stable = getsortStable();
        if(stable){
            mergeSortGPU(d_buffer, batch_size);
        }
        else bitonicSortGPU(d_buffer, batch_size);

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

    __host__ void queryKeys(const Key* keys, int size, Value* results, bool* foundFlags) {
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

    __host__ void printLevel(int level) const {
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
                    // Print index and key
                    printf("%d\t", i);
                    printValue(*(h_level[i].first)); // Custom print for key
                    printf("\t");

                    // Print value or tombstone
                    if (!h_level[i].isValueTombstone()) {
                        printValue(*(h_level[i].second)); // Custom print for value
                    } else {
                        printf("tombstone");
                    }
                    printf("\n");
                }
            }
        }

        printf("Total entries: %d/%d\n", numEntries, level_size);
        printf("----------------------------------------\n");

        delete[] h_level;
    }

    __host__ void printAllLevels() const {
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

    __host__ void countKeys(const Key* k1, const Key* k2, int numQueries, int* counts) {
        Key *d_k1, *d_k2;
        cudaMalloc(&d_k1, numQueries * sizeof(Key));
        cudaMalloc(&d_k2, numQueries * sizeof(Key));
        cudaMemcpy(d_k1, k1, numQueries * sizeof(Key), cudaMemcpyHostToDevice);
        cudaMemcpy(d_k2, k2, numQueries * sizeof(Key), cudaMemcpyHostToDevice);

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
        int findThreads = 256;
        int totalfindThreads = numQueries * numLevels;
        int findBlocks = (totalfindThreads + findThreads - 1) / findThreads;
        findBounds<<<findBlocks, findThreads>>>(d_l, d_u, d_k1, d_k2, d_init_count, bufferSize, m, numLevels, numQueries);
        cudaDeviceSynchronize();

        int* d_offset;
        cudaMalloc(&d_offset, numQueries * numLevels * sizeof(int));
        int* d_maxoffset;
        cudaMalloc(&d_maxoffset, numQueries * sizeof(int));

        int threadsPerBlock = 256;
        int blocks = (numQueries + threadsPerBlock - 1) / threadsPerBlock;
        exclusiveSum<<<blocks, threadsPerBlock>>>(d_init_count, d_offset, d_maxoffset, numQueries, numLevels);
        cudaDeviceSynchronize();
        
        int* d_maxResultSize;
        cudaMalloc(&d_maxResultSize, sizeof(int));
        int reductionThreads = 256;
        int reductionBlocks = (numQueries + reductionThreads - 1) / reductionThreads;
        reduceSum<<<reductionBlocks, reductionThreads>>>(d_maxoffset, d_maxResultSize, numQueries);
        cudaDeviceSynchronize();

        int maxResultSize;
        cudaMemcpy(&maxResultSize, d_maxResultSize, sizeof(int), cudaMemcpyDeviceToHost);

        int* d_result_offset;
        cudaMalloc(&d_result_offset, numQueries * sizeof(int));
        std::vector<int> h_maxoffset(numQueries);
        int* h_result_offset = new int[numQueries];
        cudaMemcpy(h_maxoffset.data(), d_maxoffset, numQueries * sizeof(int), cudaMemcpyDeviceToHost);

        int offset = 0;
        for(int i=0; i < numQueries; i++){
            int segmentLength = h_maxoffset[i];;
            h_result_offset[i] = offset;
            offset += segmentLength;
        }
        cudaMemcpy(d_result_offset, h_result_offset, numQueries * sizeof(int), cudaMemcpyHostToDevice);


        Pair<Key, Value>* d_result;
        cudaMalloc(&d_result, maxResultSize * sizeof(Pair<Key, Value>));
        collectElements<<<findBlocks, findThreads>>>(d_l, d_u, d_offset, d_result_offset, d_result, bufferSize, m, numLevels, numQueries);
        cudaDeviceSynchronize();

        int* d_counts;
        cudaMalloc(&d_counts, numQueries * sizeof(int));
        count<<<blocks, threadsPerBlock>>>(d_result, d_maxoffset, d_result_offset, d_k1, d_k2, d_counts, numQueries);
        cudaDeviceSynchronize();
        cudaMemcpy(counts, d_counts, numQueries * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_k1);
        cudaFree(d_k2);
        cudaFree(d_l);
        cudaFree(d_u);
        cudaFree(d_init_count);
        cudaFree(d_offset);
        cudaFree(d_maxoffset);
        cudaFree(d_result_offset);
        cudaFree(d_result);
        cudaFree(d_counts);
    }
    
    __host__ void rangeKeys(const Key* k1, const Key* k2, int numQueries, Pair<Key, Value>* range, int* counts, int* range_offset) {
        Key *d_k1, *d_k2;
        cudaMalloc(&d_k1, numQueries * sizeof(Key));
        cudaMalloc(&d_k2, numQueries * sizeof(Key));
        cudaMemcpy(d_k1, k1, numQueries * sizeof(Key), cudaMemcpyHostToDevice);
        cudaMemcpy(d_k2, k2, numQueries * sizeof(Key), cudaMemcpyHostToDevice);
        
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

        int findThreads = 256;
        int totalfindThreads = numQueries * numLevels;
        int findBlocks = (totalfindThreads + findThreads - 1) / findThreads;
        findBounds<<<findBlocks, findThreads>>>(d_l, d_u, d_k1, d_k2, d_init_count, bufferSize, m, numLevels, numQueries);
        
        int* d_offset;
        cudaMalloc(&d_offset, numQueries * numLevels * sizeof(int));
        int* d_maxoffset;
        cudaMalloc(&d_maxoffset, numQueries * sizeof(int));
        
        int threadsPerBlock = 256;
        int blocks = (numQueries + threadsPerBlock - 1) / threadsPerBlock;
        exclusiveSum<<<blocks, threadsPerBlock>>>(d_init_count, d_offset, d_maxoffset, numQueries, numLevels);
        
        int* d_maxResultSize;
        cudaMalloc(&d_maxResultSize, sizeof(int));
        int reductionThreads = 256;
        int reductionBlocks = (numQueries + reductionThreads - 1) / reductionThreads;
        reduceSum<<<reductionBlocks, reductionThreads>>>(d_maxoffset, d_maxResultSize, numQueries);
        
        int maxResultSize;
        cudaMemcpy(&maxResultSize, d_maxResultSize, sizeof(int), cudaMemcpyDeviceToHost);
        
        int* d_result_offset;
        cudaMalloc(&d_result_offset, numQueries * sizeof(int));
        std::vector<int> h_maxoffset(numQueries);
        int* h_result_offset = new int[numQueries];
        cudaMemcpy(h_maxoffset.data(), d_maxoffset, numQueries * sizeof(int), cudaMemcpyDeviceToHost);

        int offset = 0;
        for(int i=0; i < numQueries; i++){
            int segmentLength = h_maxoffset[i];;
            h_result_offset[i] = offset;
            offset += segmentLength;
        }
        cudaMemcpy(d_result_offset, h_result_offset, numQueries * sizeof(int), cudaMemcpyHostToDevice);


        Pair<Key, Value>* d_result;
        cudaMalloc(&d_result, maxResultSize * sizeof(Pair<Key, Value>));
        collectElements<<<findBlocks, findThreads>>>(d_l, d_u, d_offset, d_result_offset, d_result, bufferSize, m, numLevels, numQueries);

        sortBySegment(d_result, d_maxoffset, numQueries);
        
        int* d_counts;
        cudaMalloc(&d_counts, numQueries * sizeof(int));
        Pair<Key, Value>* d_range;
        cudaMalloc(&d_range, maxResultSize * sizeof(Pair<Key, Value>));
        compact<<<blocks, threadsPerBlock>>>(d_result, d_maxoffset, d_result_offset, d_range, d_counts, numQueries);
        
        cudaMemcpy(range, d_range, maxResultSize * sizeof(Pair<Key, Value>), cudaMemcpyDeviceToHost);
        cudaMemcpy(counts, d_counts, numQueries * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(range_offset, d_result_offset, numQueries * sizeof(int), cudaMemcpyDeviceToHost);
        
        cudaFree(d_k1);
        cudaFree(d_k2);
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

};

#endif // GPU_LSM_TREE_H



