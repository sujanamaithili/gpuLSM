#ifndef GPU_MERGE_SORT_HELPER_H
#define GPU_MERGE_SORT_HELPER_H

#include "lsm.cuh"
#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>

template <typename Key, typename Value>
__device__ void Merge(Pair<Key, Value>* arr, Pair<Key, Value>* temp, long int left, long int middle, long int right) {
    long int i = left, j = middle, k = left;
    while (i < middle && j < right) {
        if (arr[i].first <= arr[j].first)
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }
    while (i < middle) temp[k++] = arr[i++];
    while (j < right) temp[k++] = arr[j++];
    for (long int x = left; x < right; x++) arr[x] = temp[x];
}

// Kernel function for merge sort
template <typename Key, typename Value>
__global__ void mergeSortKernel(Pair<Key, Value>* arr, Pair<Key, Value>* temp, long int n, long int width) {
    long int tid = blockIdx.x * blockDim.x + threadIdx.x;
    long int left = tid * width;
    long int middle = min(left + width / 2, n);
    long int right = min(left + width, n);
    if (left < n && middle < n) {
        Merge(arr, temp, left, middle, right);
    }
}

// Host function for GPU merge sort with internal configuration
template <typename Key, typename Value>
void mergeSortGPU(Pair<Key, Value>* d_arr, long int n) {
    const int threadsPerBlock = 256;  // Set a typical number of threads per block
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate temporary array on the GPU
    Pair<Key, Value>* d_temp;
    cudaMalloc(&d_temp, n * sizeof(Pair<Key, Value>));

    for (long int width = 1; width < n; width *= 2) {
        mergeSortKernel<Key, Value><<<blocksPerGrid, threadsPerBlock>>>(d_arr, d_temp, n, 2 * width);
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_temp);
            return;
        }
    }

    // Free the temporary array
    cudaFree(d_temp);
}

template <typename Key, typename Value>
void mergeCPU(Pair<Key, Value>* arr, Pair<Key, Value>* temp, long int left, long int middle, long int right) 
{
    long int i = left, j = middle, k = left;
    while (i < middle && j < right) 
    {
        if (arr[i].first <= arr[j].first)
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }

    while (i < middle)
        temp[k++] = arr[i++];
    while (j < right)
        temp[k++] = arr[j++];

    for (long int x = left; x < right; x++)
        arr[x] = temp[x];
}

template <typename Key, typename Value>
void mergeSortCPU(Pair<Key, Value>* arr, Pair<Key, Value>* temp, long int left, long int right) 
{
    if (right - left <= 1)
        return;
    long int middle = (left + right) / 2;
    mergeSortCPU(arr, temp, left, middle);
    mergeSortCPU(arr, temp, middle, right);
    mergeCPU(arr, temp, left, middle, right);
}

template <typename Key, typename Value>
void sortBySegment(Pair<Key, Value>* d_result, int* d_maxoffset, int numQueries) {
    std::vector<int> h_maxoffset(numQueries);
    cudaMemcpy(h_maxoffset.data(), d_maxoffset, numQueries * sizeof(int), cudaMemcpyDeviceToHost);

    int offset = 0;
    for(int i=0; i < numQueries; i++){
        int segmentLength = h_maxoffset[i];;
        
        mergeSortGPU(d_result + offset, segmentLength);

        offset += segmentLength;
    }
    
}

// Function to check if the array is sorted
template <typename Key, typename Value>
bool isSorted2(Pair<Key, Value>* arr, long int n) {
    for (long int i = 1; i < n; i++) {
        if (arr[i - 1].first > arr[i].first) return false;
    }
    return true;
}

#endif 
