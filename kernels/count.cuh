#ifndef GPU_COUNT_HELPER_H
#define GPU_COUNT_HELPER_H

#include "lsm.cuh"

template <typename Key, typename Value>
__global__ void count(const Pair<Key, Value>* d_result, const int* d_maxoffset, const int* d_result_offset, const Key* d_k1, const Key* d_k2, int* d_counts, int numQueries) {
    int queryId = blockIdx.x * blockDim.x + threadIdx.x;
    if (queryId >= numQueries) return;
    Key sk = d_k1[queryId];
    Key ek = d_k2[queryId];
    int segmentLength = d_maxoffset[queryId];
    int validCount = 0;

    if(segmentLength > 0){
        int segmentStart = d_result_offset[queryId];

        for (Key currentKey = sk; currentKey <= ek; ++currentKey) {
            for (int i = 0; i < segmentLength; i++) {
                // Locate the first occurrence of the current key
                if (d_result[segmentStart + i].first == currentKey) {
                    // Check the value validity
                    if (!d_result[segmentStart + i].isValueTombstone()) {
                        validCount++;
                    }
                    // Break the loop after the first occurrence is found
                    break;
                }
            }
        }
    }

    d_counts[queryId] = validCount;

} 



#endif