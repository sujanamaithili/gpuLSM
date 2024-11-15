#include <lsm.cuh>
#include <query.cuh>
#include <cuda.h>
#include <cstdio>

template <typename Key, typename Value>
__device__ bool binarySearchFirstOccurrence(const Pair<Key, Value>* data, int size, Key key, Value& value) {
    int left = 0;
    int right = size - 1;
    int resultIndex = -1;

    while (left <= right) {
        int mid = left + ((right - left) >> 1);
        Key mid_key = data[mid].first;

        if (mid_key == key) {
            resultIndex = mid;
            right = mid - 1;
        } else if (mid_key < key) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    if (resultIndex != -1) {
        value = data[resultIndex].second;
        return true;
    } else {
        return false;
    }
}



template <typename Key, typename Value>
__global__ void queryKeysKernel(const Key* d_keys, Value* d_results, bool* d_foundFlags, int size, const Pair<Key, Value>* d_memory, int num_levels, int buffer_size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size) return;

    Key key = d_keys[i];
    bool found = false;
    Value value;
    int offset = 0;

    for (int level = 0; level < num_levels; ++level) {
        int level_size = buffer_size << level;
        const Pair<Key, Value>* level_data = d_memory + offset;
        if (binarySearchFirstOccurrence(level_data, level_size, key, value)) {
            found = true;
            break;
        }
        offset += level_size;
    }

    if (found) {
        d_results[i] = value;
        d_foundFlags[i] = true;
    } else {
        d_foundFlags[i] = false;
    }
}

//TODO : Check if this is required ? Explicit template instantiation for int, int
template __global__ void queryKeysKernel<int, int>(const int*, int*, bool*, int, const Pair<int, int>*, int, int);
