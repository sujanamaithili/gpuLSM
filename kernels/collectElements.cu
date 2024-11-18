#include <cuda_runtime.h>

#include <collectElements.cuh>

template <typename Key, typename Value>
__global__ void collectElements(const int* d_l, const int* d_u, const int* d_offset, const int* d_result_offset, Pair<Key, Value>* d_result, int bufferSize, Pair<Key, Value>* m, int numLevels) {
    int queryId = blockIdx.x;
    int level = threadIdx.x;

    int offset = 0;
    for (int i = 0; i < level; ++i) {
        offset += bufferSize << i;  
    }

    Pair<Key, Value>* levelData =  m + offset;

    int startIdx = d_offset[queryId * numLevels + level];
    if(queryId > 0){
        startIdx += d_result_offset[queryId];
    }

    int lower = d_l[queryId * numLevels + level];
    int upper = d_u[queryId * numLevels + level];
    int level_size = bufferSize << level; 

    // Collect elements within the range and store them in result buffer
    for (int i = lower; i < upper && i < level_size; ++i) {
        d_result[startIdx + (i - lower)] = levelData[i];  
    }
}

template __global__ void collectElements<int, int>(const int*, const int*, const int*, const int*, Pair<int, int>*, int, Pair<int, int>*, int);
