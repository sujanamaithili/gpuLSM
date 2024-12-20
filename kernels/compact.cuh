#ifndef GPU_COMPACT_HELPER_H
#define GPU_COMPACT_HELPER_H

#include "lsm.cuh"

template <typename Key, typename Value>
__global__ void compact(const Pair<Key, Value>* d_result, const int* d_maxoffset, const int* d_result_offset, Pair<Key, Value>* d_range, int* d_counts, int numQueries) {
    int queryId = blockIdx.x * blockDim.x + threadIdx.x;
    if (queryId >= numQueries) return;

    int segmentLength = d_maxoffset[queryId];
    int validCount = 0;

    if(segmentLength > 0){
        int segmentStart = d_result_offset[queryId];
        std::optional<Key> lastKey = d_result[segmentStart].first;
	if(!d_result[segmentStart].isValueTombstone()){
            d_range[segmentStart] = d_result[segmentStart];
            validCount = 1;
        }
        

        for (int i = 1; i < segmentLength; i++) {
            if (d_result[segmentStart + i].first != lastKey) {
                lastKey = d_result[segmentStart + i].first;
                if(!d_result[segmentStart + i].isValueTombstone()){
                    d_range[segmentStart + validCount] = d_result[segmentStart + i];
                    validCount++;
                }
                
            }
        }
    }

    d_counts[queryId] = validCount;

}  

#endif
