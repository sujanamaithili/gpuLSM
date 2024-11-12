#ifndef MERGE_SORT_CUH
#define MERGE_SORT_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define MAX_THREADS_PER_BLOCK 1024

// Device function for merging two sorted subarrays
__device__ void Merge(int* arr, int* temp, int left, int middle, int right);

// Kernel function for merge sort on GPU
__global__ void MergeSortGPU(int* arr, int* temp, int n, int width);

// Host function to perform merge sort on the GPU
void mergeSort(int* arr, int n);

#endif // MERGE_SORT_CUH
