#include <lsm.cuh>
#include <query.cuh>
#include <cuda.h>
#include <cstdio>
#include <optional>

template <typename Key, typename Value>
__device__ bool binarySearchFirstOccurrence(const Pair<Key, Value>* data, int size, Key key, Value& value) {
    int left = 0;
    int right = size - 1;
    int resultIndex = -1;

    while (left <= right) {
        int mid = left + ((right - left) >> 1);
        
        if (data[mid].isKeyEmpty()) {
            right = mid - 1;
            continue;
        }

        const Key& mid_key = *(data[mid].first);

        if (mid_key == key) {
            resultIndex = mid;
            right = mid - 1;
        } else if (mid_key < key) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    if (resultIndex != -1 && !data[resultIndex].isValueTombstone()) {
        value = *(data[resultIndex].second);
        return true;
    }
    return false;
}

template <typename Key, typename Value>
__global__ void queryKeysKernel(const Key* d_keys, Value* d_results, bool* d_foundFlags, int size, const Pair<Key, Value>* d_memory, int num_levels, int buffer_size, int num_batches) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size) return;

    Key key = d_keys[i];
    bool found = false;
    Value value;
    int offset = 0;

    // Search through each level
    for (int level = 0; level < num_levels && !found; ++level) {

        if (!(num_batches & (1 << level))) {
            offset += buffer_size << level; 
            continue;
        }

        const int level_size = buffer_size << level;
        const Pair<Key, Value>* level_data = d_memory + offset;   
        if (binarySearchFirstOccurrence(level_data, level_size, key, value)) {
            found = true;
            d_results[i] = value;
            break;
        }
        offset += level_size;
    }

    d_foundFlags[i] = found;
}

// Explicit template instantiation
template __global__ void queryKeysKernel<int, int>(
    const int*, int*, bool*, int, const Pair<int, int>*, int, int, int);
