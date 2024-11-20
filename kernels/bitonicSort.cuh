#ifndef GPU_BITONIC_SORT_HELPER_H
#define GPU_BITONIC_SORT_HELPER_H

#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include "lsm.cuh"

template <typename Key, typename Value>
__global__ void bitonicSortKernel(Pair<Key, Value>* arr, long int j, long int k) {
    long int i = blockIdx.x * blockDim.x + threadIdx.x;
    long int ij = i ^ j;

    if (ij > i) {
        bool isNullOpt1 = !arr[i].second.has_value();
        bool isNullOpt2 = !arr[ij].second.has_value();

        if ((i & k) == 0) {
            if ((!isNullOpt1 && isNullOpt2 && arr[i].first == arr[ij].first) || (arr[i].first > arr[ij].first)) {
                Pair<Key, Value> temp = arr[i];
                arr[i] = arr[ij];
                arr[ij] = temp;
            }
        }
        else {
            if ((isNullOpt1 && !isNullOpt2 && arr[i].first == arr[ij].first) ||  (arr[i].first < arr[ij].first)){ 
                Pair<Key, Value> temp = arr[i];
                arr[i] = arr[ij];
                arr[ij] = temp;
            }
        }
    }
}

template <typename Key, typename Value>
void bitonicSortGPU(Pair<Key, Value>* d_arr, long int n) {
    const int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    for (long int k = 2; k <= n; k *= 2) {
        for (long int j = k / 2; j > 0; j /= 2) {
            bitonicSortKernel<Key, Value><<<blocksPerGrid, threadsPerBlock>>>(d_arr, j, k);
            cudaDeviceSynchronize();

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
                return;
            }
        }
    }
}

template <typename Key, typename Value>
void bitonicSortCPU(Pair<Key, Value>* arr, long int n) {
    for (long int k = 2; k <= n; k *= 2) {
        for (long int j = k / 2; j > 0; j /= 2) {
            for (long int i = 0; i < n; i++) {
                long int ij = i ^ j;
                if (ij > i) {
                    bool isNullOpt1 = !arr[i].second.has_value();
                    bool isNullOpt2 = !arr[ij].second.has_value();

                    if ((i & k) == 0) {
                        // Ascending order: nullopt has the highest priority
                        if (isNullOpt1 || (!isNullOpt2 && arr[i].first > arr[ij].first)) {
                            Pair<Key, Value> temp = arr[i];
                            arr[i] = arr[ij];
                            arr[ij] = temp;
                        }
                    } else {
                        // Descending order: nullopt has the highest priority
                        if (!isNullOpt1 && (isNullOpt2 || arr[i].first < arr[ij].first)) {
                            Pair<Key, Value> temp = arr[i];
                            arr[i] = arr[ij];
                            arr[ij] = temp;
                        }
                    }
                }
            }
        }
    }
}

#endif 
