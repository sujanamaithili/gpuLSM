#include <cuda_runtime.h>
#include "gpu_merge_helper.h"

template <typename Key, typename Value>
__global__ void mergeKernel(Pair<Key, Value>* d_arr1, int size1, Pair<Key, Value>* d_arr2, int size2, Pair<Key, Value>* d_merged) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < (size1 + size2)) {

    }
}


template <typename Key, typename Value>
__host__ Pair<Key, Value>* merge(Pair<Key, Value>* d_arr1, int size1, Pair<Key, Value>* d_arr2, int size2) {
    Pair<Key, Value>* d_merged;
    cudaMalloc(&d_merged, (size1 + size2) * sizeof(Pair<Key, Value>));

    //TODO: get configurations through command line
    int threadsPerBlock = 256;
    int blocksPerGrid = (size1 + size2 + threadsPerBlock - 1) / threadsPerBlock;
    
    mergeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr1, size1, d_arr2, size2, d_merged);
    cudaDeviceSynchronize();

    return d_merged;
}
