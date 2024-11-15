#include <iostream>
#include <cuda.h>
#include <cstdlib>
#include <algorithm>
#include <time.h>

__device__ void Merge(int* arr, int* temp, long int left, long int middle, long int right) 
{
    long int i = left, j = middle, k = left;
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

    for (long int x = left; x < right; x++)
        arr[x] = temp[x];
}

__global__ void MergeSortGPU(int* arr, int* temp, long int n, long int width) 
{
    long int tid = threadIdx.x + blockDim.x * blockIdx.x;
    long int left = tid * width;
    long int middle = min(left + width / 2, n);
    long int right = min(left + width, n);
    if (left < n && middle < n) 
    {
        Merge(arr, temp, left, middle, right);
    }
}

void mergeSortGPU(int* d_arr, int* d_temp, long int n, int threadsPerBlock) 
{
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    for (long int wid = 1; wid < n; wid *= 2) 
    {
        MergeSortGPU<<<blocksPerGrid, threadsPerBlock>>>(d_arr, d_temp, n, 2 * wid);
    }
}

void mergeCPU(int* arr, int* temp, long int left, long int middle, long int right) 
{
    long int i = left, j = middle, k = left;
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

    for (long int x = left; x < right; x++)
        arr[x] = temp[x];
}

void mergeSortCPU(int* arr, int* temp, long int left, long int right) 
{
    if (right - left <= 1)
        return;
    long int middle = (left + right) / 2;
    mergeSortCPU(arr, temp, left, middle);
    mergeSortCPU(arr, temp, middle, right);
    mergeCPU(arr, temp, left, middle, right);
}

bool isSorted(int* arr, long int n) 
{
    for (long int i = 1; i < n; i++) 
    {
        if (arr[i - 1] > arr[i])
            return false;
    }
    return true;
}

int main() 
{
    const long int N = 1L << exponent;
    int* arrGPU = new int[N];
    int* arrCPU = new int[N];
    int* temp = new int[N];

    for (long int i = 0; i < N; i++) 
    {
        int value = rand() % N;
        arrGPU[i] = value;
        arrCPU[i] = value;
    }

    // Allocate memory on the GPU
    int* d_arr;
    int* d_temp;
    long int size = N * sizeof(int);
    cudaMalloc(&d_arr, size);
    cudaMalloc(&d_temp, size);

    // Perform GPU merge sort
    clock_t start, end;
    cudaMemcpy(d_arr, arrGPU, size, cudaMemcpyHostToDevice);
    start = clock();
    mergeSortGPU(d_arr, d_temp, N, threadsPerBlock);
    cudaDeviceSynchronize();
    end = clock();
    double gpuTime = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken in GPU = %lf s\n", gpuTime);

    // Copy the sorted array back to host (not timed)
    cudaMemcpy(arrGPU, d_arr, size, cudaMemcpyDeviceToHost);

    // Measure CPU merge sort time
    start = clock();
    mergeSortCPU(arrCPU, temp, 0, N);
    end = clock();
    double cpuTime = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken in CPU = %lf s\n", cpuTime);

    // Verify if the arrays are sorted
    bool gpuSorted = isSorted(arrGPU, N);
    bool cpuSorted = isSorted(arrCPU, N);
    std::cout << "GPU Merge Sort: " << (gpuSorted ? "Sorted correctly" : "Not sorted correctly") << std::endl;
    std::cout << "CPU Merge Sort: " << (cpuSorted ? "Sorted correctly" : "Not sorted correctly") << std::endl;

    // Clean up
    delete[] arrGPU;
    delete[] arrCPU;
    delete[] temp;
    cudaFree(d_arr);
    cudaFree(d_temp);

    return 0;
}
