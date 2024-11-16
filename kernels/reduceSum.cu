#include <cuda_runtime.h>

#include "reduceSum.cuh"

__global__ void reduceSum(const int* input, int* result, int size) {
    __shared__ int sharedData[256]; 

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input elements into shared memory
    sharedData[tid] = (idx < size) ? input[idx] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sharedData[0]);
    }
}