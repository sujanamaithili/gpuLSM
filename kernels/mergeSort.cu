#include <iostream>
#include <cuda.h>
#include <cstdlib>   
#include <algorithm> 
#include <time.h>

#define MAX_THREADS_PER_BLOCK 1024

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

void mergeSortGPU(int* arr, int n) 
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
    }

    cudaMemcpy(arr, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_temp);
}

void mergeCPU(int* arr, int* temp, int left, int middle, int right) 
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

void mergeSortCPU(int* arr, int* temp, int left, int right) 
{
    if (right - left <= 1)
        return;

    int middle = (left + right) / 2;
    mergeSortCPU(arr, temp, left, middle);
    mergeSortCPU(arr, temp, middle, right);
    mergeCPU(arr, temp, left, middle, right);
}

bool isSorted(int* arr, int n) 
{
    for (int i = 1; i < n; i++) 
    {
        if (arr[i - 1] > arr[i])
            return false;
    }
    return true;
}

int main() 
{
    const int N = 1 << 23;
    int* arrGPU = new int[N];
    int* arrCPU = new int[N];
    int* temp = new int[N];

    for (int i = 0; i < N; i++) 
    {
        int value = rand() % N;
        arrGPU[i] = value;
        arrCPU[i] = value;
    }

    clock_t start, end;
    double time_taken;
    // Measure GPU merge sort time
    start = clock();
    mergeSortGPU(arrGPU, N);
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("Time taken in GPU = %lf\n", time_taken);

    // Measure CPU merge sort time
    start = clock();
    mergeSortCPU(arrCPU, temp, 0, N);
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("Time taken in CPU = %lf\n", time_taken);

    // Check if the arrays are sorted
    bool gpuSorted = isSorted(arrGPU, N);
    bool cpuSorted = isSorted(arrCPU, N);

    std::cout << "GPU Merge Sort: " << (gpuSorted ? "Sorted correctly" : "Not sorted correctly") <<  std::endl;
    std::cout << "CPU Merge Sort: " << (cpuSorted ? "Sorted correctly" : "Not sorted correctly") <<  std::endl;

    // Clean up
    delete[] arrGPU;
    delete[] arrCPU;
    delete[] temp;

    return 0;
}
