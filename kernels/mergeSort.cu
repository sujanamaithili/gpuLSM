#include "mergeSort.cuh"
#include <algorithm> 
#include <cstdlib>   
#include <iostream>

// Device function for merging two sorted subarrays
__device__ void Merge(int* arr, int* temp, int left, int middle, int right) 
{
    int i = left, j = middle, k = left;

    while (i < middle && j < right) 
    {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }

    while (i < middle)
        temp[k++] = arr[i++];
    while (j < right)
        temp[k++] = arr[j++];

    for (int x = left; x < right; x++)
        arr[x] = temp[x];
}

// Kernel function for merge sort on GPU
__global__ void MergeSortGPU(int* arr, int* temp, int n, int width) 
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int left = tid * width;
    int middle = min(left + width / 2, n);
    int right = min(left + width, n);

    if (left < n && middle < n) 
    {
        Merge(arr, temp, left, middle, right);
    }
}

// Host function to perform merge sort on the GPU
void mergeSort(int* arr, int n) 
{
    int* d_arr;
    int* d_temp;
    size_t size = n * sizeof(int);

    cudaMalloc(&d_arr, size);
    cudaMalloc(&d_temp, size);
    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = MAX_THREADS_PER_BLOCK;

    for (int wid = 1; wid < n; wid *= 2) 
    {
        int numTasks = (n + 2 * wid - 1) / (2 * wid);
        int blocksPerGrid = (numTasks + threadsPerBlock - 1) / threadsPerBlock;

        MergeSortGPU<<<blocksPerGrid, threadsPerBlock>>>(d_arr, d_temp, n, 2 * wid);

        cudaDeviceSynchronize();
    }

    cudaMemcpy(arr, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_temp);
}
