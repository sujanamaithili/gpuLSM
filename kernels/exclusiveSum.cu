#include <cuda_runtime.h>

#include <exclusiveSum.cuh>

__global__ void exclusiveSum(const int* d_init_count, int* d_offset, int* d_maxoffset, int numQueries, int numLevels) {
    
    int queryId = blockIdx.x * blockDim.x + threadIdx.x;

    if(queryId < numQueries) {
        int sum = 0;
        int numLevels = numLevels;
        for (int level = 0; level < numLevels; ++level) {
            int idx = queryId * numLevels + level;
            d_offset[idx] = sum; 
            sum += d_init_count[idx];          
        }
        d_maxoffset[queryId] = sum;
    }

}
