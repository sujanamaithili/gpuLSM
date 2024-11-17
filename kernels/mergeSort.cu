#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include "mergeSort.cuh"

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
void sortBySegment(Pair<Key, Value>* d_result, int* d_maxoffset, int* d_result_offset, int numQueries) {
    std::vector<int> h_maxoffset(numQueries);
    cudaMemcpy(h_maxoffset.data(), d_maxoffset, numQueries * sizeof(int), cudaMemcpyDeviceToHost);

    int offset = 0;
    for(int i=0; i < numQueries; i++){
        int segmentLength = h_maxoffset[i];;
        
        mergeSortGPU(d_result + offset, segmentLength);

        d_result_offset[i] = offset;

        offset += segmentLength;
    }
}

// Function to check if the array is sorted
template <typename Key, typename Value>
bool isSorted(Pair<Key, Value>* arr, long int n) {
    for (long int i = 1; i < n; i++) {
        if (arr[i - 1].first > arr[i].first) return false;
    }
    return true;
}

void testMergeSort() {
    const long int N = 1L << 20;  // 1 million elements
    using Key = int;
    using Value = int;
    using DataType = Pair<Key, Value>;

    // Allocate host memory
    DataType* arrCPU = new DataType[N];
    DataType* arrGPU = new DataType[N];
    DataType* temp = new DataType[N];

    // Initialize the arrays with random key-value pairs
    for (long int i = 0; i < N; i++) {
        int randomValue = rand() % N;
        arrCPU[i] = {randomValue, (Key)i};
        arrGPU[i] = {randomValue, (Key)i};
    }

    // Allocate device memory
    DataType* d_arr;
    cudaMalloc(&d_arr, N * sizeof(DataType));

    // Copy data to device
    cudaMemcpy(d_arr, arrGPU, N * sizeof(DataType), cudaMemcpyHostToDevice);

    // Measure GPU merge sort time using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mergeSortGPU<Key, Value>(d_arr, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuTime;
    cudaEventElapsedTime(&gpuTime, start, stop);
    std::cout << "Time taken in GPU merge sort = " << gpuTime / 1000.0 << " s\n";

    // Copy sorted data back to host
    cudaMemcpy(arrGPU, d_arr, N * sizeof(DataType), cudaMemcpyDeviceToHost);

    // Measure CPU merge sort time
    clock_t cpuStart = clock();
    mergeSortCPU(arrCPU, temp, 0, N);
    clock_t cpuEnd = clock();
    double cpuTime = (double)(cpuEnd - cpuStart) / CLOCKS_PER_SEC;
    std::cout << "Time taken in CPU merge sort = " << cpuTime << " s\n";

    // Verify if the arrays are sorted
    bool gpuSorted = isSorted(arrGPU, N);
    bool cpuSorted = isSorted(arrCPU, N);
    std::cout << "GPU Merge Sort: " << (gpuSorted ? "Sorted correctly" : "Not sorted correctly") << std::endl;
    std::cout << "CPU Merge Sort: " << (cpuSorted ? "Sorted correctly" : "Not sorted correctly") << std::endl;

    // Clean up resources
    delete[] arrGPU;
    delete[] arrCPU;
    delete[] temp;
    cudaFree(d_arr);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

template void sortBySegment<int, int>(Pair<int, int>* , int* , int* , int);
// int main() {
//     testMergeSort();
//     return 0;
// }
