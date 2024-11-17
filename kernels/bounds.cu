#include <cuda_runtime.h>

#include <bounds.cuh>

template <typename Key, typename Value>
__device__ int lowerBound(int level, Key key, int bufferSize, Pair<Key, Value>* m) {
    int offset = 0;
    int level_size = bufferSize << level;

    for (int i = 0; i < level; ++i) {
        offset += bufferSize << i;  
    }

    Pair<Key, Value>* levelData =  m + offset;

    int left = 0;
    int right = level_size;

    while (left < right) {
        int mid = left + (right - left) / 2;
        if (levelData[mid].first < key) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return left;
}

template <typename Key, typename Value>
__device__ int upperBound(int level, Key key, int bufferSize, Pair<Key, Value>* m) {
    int offset = 0;
    int level_size = bufferSize << level;

    for (int i = 0; i < level; ++i) {
        offset += bufferSize << i;  
    }

    Pair<Key, Value>* levelData =  m + offset;

    int left = 0;
    int right = level_size;

    while (left < right) {
        int mid = left + (right - left) / 2;
        if (levelData[mid].first <= key) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return left;
}




template <typename Key>
__global__ void findBounds(int* d_l, int* d_u, const Key* k1, const Key* k2, int* d_init_count, int bufferSize, Pair<Key, Value>* m, int numLevels) {
    int queryId = blockIdx.x;
    int level = threadIdx.x;

    Key key1 = k1[queryId];
    Key key2 = k2[queryId];

    d_l[queryId * numLevels + level] = lowerBound(level, key1, bufferSize, m);
    d_u[queryId * numLevels + level] = upperBound(level, key2, bufferSize, m);

    d_init_count[queryId * numLevels + level] = d_u[queryId * numLevels + level] - d_l[queryId * numLevels + level];

}

template __global__ void findBounds<int, int>( int*, int*, const Key*, const Key*, int*, int);

