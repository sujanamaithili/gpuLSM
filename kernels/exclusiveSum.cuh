#ifndef GPU_EXCLUSIVE_SUM_HELPER_H
#define GPU_EXCLUSIVE_SUM_HELPER_H

#include "lsm.cuh"

__global__ void exclusiveSum(const int* d_init_count, int* d_offset, int* d_maxoffset, int numQueries, int numLevels) {
    
    int queryId = blockIdx.x * blockDim.x + threadIdx.x;

    if(queryId < numQueries) {
        int sum = 0;
        for (int level = 0; level < numLevels; ++level) {
            int idx = queryId * numLevels + level;
            d_offset[idx] = sum; 
            sum += d_init_count[idx];          
        }
        d_maxoffset[queryId] = sum;
    }

}

#endif