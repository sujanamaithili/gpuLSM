#ifndef GPU_MERGE_HELPER_H
#define GPU_MERGE_HELPER_H

#include <cuda.h>
#include <iostream>
#include <vector>
#include <optional>
#include <algorithm>
#include <cassert>
#include "lsm.cuh"

template <typename Key, typename Value>
__global__ void mergeKernel(Pair<Key, Value>* d_arr1, int size1, Pair<Key, Value>* d_arr2, int size2, Pair<Key, Value>* d_merged) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if( idx >= size1 + size2 ) return;

    int start1 = 0;
    int start2 = 0;
    int k = idx+1;

    while(true){
        // If arr1 is exhausted, return the k-th element from arr2
        if(start1 == size1){
            d_merged[idx] = d_arr2[start2 + k - 1];
            return;
        }

        // If arr2 is exhausted, return the k-th element from arr1
        if (start2 == size2) {
            d_merged[idx] = d_arr1[start1 + k - 1];
            return;
        }

        // If k == 1, return the minimum of the first elements in both arrays
        if (k == 1) {
            d_merged[idx] = (d_arr1[start1].first <= d_arr2[start2].first) ? d_arr1[start1] : d_arr2[start2];
            return;
        }

        // Calculate midpoints in the remaining portions of arr1 and arr2
        int mid1 = min(size1 - start1, k / 2);
        int mid2 = min(size2 - start2, k / 2);

        if (d_arr1[start1 + mid1 - 1].first <= d_arr2[start2 + mid2 - 1].first) {
            // Discard the first 'mid1' elements of d_arr1
            start1 += mid1;
            k -= mid1;
        } else {
            // Discard the first 'mid2' elements of d_arr2
            start2 += mid2;
            k -= mid2;
        }

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

template <typename Key, typename Value>
std::vector<Pair<Key, Value>> mergeCPU(const std::vector<Pair<Key, Value>>& arr1, const std::vector<Pair<Key, Value>>& arr2) {
    std::vector<Pair<Key, Value>> merged;
    int i = 0, j = 0;
    int size1 = arr1.size();
    int size2 = arr2.size();

    merged.reserve(size1 + size2);

    while (i < size1 && j < size2) {
        if (arr1[i].first <= arr2[j].first) {
            merged.push_back(arr1[i++]);
        } else {
            merged.push_back(arr2[j++]);
        }
    }

    while (i < size1) merged.push_back(arr1[i++]);
    while (j < size2) merged.push_back(arr2[j++]);

    return merged;
}

template <typename Key, typename Value>
bool compareResults(const std::vector<Pair<Key, Value>>& cpu_result, Pair<Key, Value>* d_gpu_result, int size) {
    // Copy GPU results back to host
    std::vector<Pair<Key, Value>> gpu_result(size);
    cudaMemcpy(gpu_result.data(), d_gpu_result, size * sizeof(Pair<Key, Value>), cudaMemcpyDeviceToHost);

    // Compare CPU and GPU results element-wise
    for (int i = 0; i < size; ++i) {
        if (cpu_result[i].first != gpu_result[i].first || cpu_result[i].second != gpu_result[i].second) {
            return false;
        }
    }
    return true;
}

void testMerge() {
    using Key = int;
    using Value = int;

    const int size1 = 5;
    const int size2 = 5;

    Pair<Key, Value> h_arr1[size1] = {{1, 10}, {3, 30}, {5, 50}, {7, 70}, {9, 90}};
    Pair<Key, Value> h_arr2[size2] = {{2, 20}, {4, 40}, {6, 60}, {8, 80}, {10, 100}};
    Pair<Key, Value> h_merged[size1 + size2];

    // Allocate device memory and copy data from host to device
    Pair<Key, Value> *d_arr1, *d_arr2;
    cudaMalloc(&d_arr1, size1 * sizeof(Pair<Key, Value>));
    cudaMalloc(&d_arr2, size2 * sizeof(Pair<Key, Value>));
    cudaMemcpy(d_arr1, h_arr1, size1 * sizeof(Pair<Key, Value>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr2, h_arr2, size2 * sizeof(Pair<Key, Value>), cudaMemcpyHostToDevice);

    // Call the merge function
    Pair<Key, Value> *d_merged = merge(d_arr1, size1, d_arr2, size2);

    // Copy the result back to the host
    cudaMemcpy(h_merged, d_merged, (size1 + size2) * sizeof(Pair<Key, Value>), cudaMemcpyDeviceToHost);

    // Print the merged array
    std::cout << "Merged array:\n";
    for (int i = 0; i < size1 + size2; i++) {
       std::cout << "(" 
          << (h_merged[i].first.has_value() ? std::to_string(h_merged[i].first.value()) : "nullopt")
          << ", " 
          << (h_merged[i].second.has_value() ? std::to_string(h_merged[i].second.value()) : "nullopt")
          << ") ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_arr1);
    cudaFree(d_arr2);
    cudaFree(d_merged);
}
#endif  // GPU_MERGE_HELPER_H
