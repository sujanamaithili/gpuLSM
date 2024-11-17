#include <cuda_runtime.h>

#include <count.cuh>

template <typename Key, typename Value>
__global__ void count(const Pair<Key, Value>* d_result, const int* d_maxoffset, const int* d_result_offset, int* d_counts, int numQueries) {
    int queryId = blockIdx.x * blockDim.x + threadIdx.x;
    if (queryId >= numQueries) return;

    int segmentLength = d_maxoffset[queryId];
    int validCount = 0;

    if(segmentLength > 0){
        int segmentStart = d_result_offset[queryId];
        std::optional<Key> lastKey = d_result[segmentStart].first;
        if(!d_result[segmentStart].isValueTombstone()){
            validCount = 1;
        }
        

        for (int i = 1; i < segmentLength; i++) {
            if (d_result[segmentStart + i].first != lastKey) {
                lastKey = d_result[segmentStart + i].first;
                if(!d_result[segmentStart + i].isValueTombstone()){
                    validCount++;
                }
            }
        }
    }

    d_counts[queryId] = validCount;

}   

template __global__ void count<int, int>(const Pair<int, int>*, const int*, const int*, int*, int);
