#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include "bitonicSort.cuh"
#include <vector>

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

template <typename Key, typename Value>
bool isSorted(Pair<Key, Value>* arr, long int n) {
    for (long int i = 1; i < n; i++) {
        if (arr[i - 1].first > arr[i].first) return false;
    }
    return true;
}

void testBitonicSort() {
    const long int N = 1L << 20;  // 1 million elements
    using Key = int;
    using Value = int;
    using DataType = Pair<Key, Value>;

    // Allocate host memory
    DataType* arrCPU = new DataType[N];
    DataType* arrGPU = new DataType[N];

    // Initialize arrays with random key-value pairs
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

    // Measure GPU bitonic sort time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    bitonicSortGPU<Key, Value>(d_arr, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuTime;
    cudaEventElapsedTime(&gpuTime, start, stop);
    std::cout << "Time taken in GPU bitonic sort = " << gpuTime / 1000.0 << " s\n";

    // Copy sorted data back to host
    cudaMemcpy(arrGPU, d_arr, N * sizeof(DataType), cudaMemcpyDeviceToHost);

    // Measure CPU bitonic sort time
    clock_t cpuStart = clock();
    bitonicSortCPU(arrCPU, N);
    clock_t cpuEnd = clock();
    double cpuTime = (double)(cpuEnd - cpuStart) / CLOCKS_PER_SEC;
    std::cout << "Time taken in CPU bitonic sort = " << cpuTime << " s\n";

    // Verify if arrays are sorted
    bool gpuSorted = isSorted(arrGPU, N);
    bool cpuSorted = isSorted(arrCPU, N);
    std::cout << "GPU Bitonic Sort: " << (gpuSorted ? "Sorted correctly" : "Not sorted correctly") << std::endl;
    std::cout << "CPU Bitonic Sort: " << (cpuSorted ? "Sorted correctly" : "Not sorted correctly") << std::endl;

    // Clean up resources
    delete[] arrGPU;
    delete[] arrCPU;
    cudaFree(d_arr);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// int main() {
//     testBitonicSort();
//     return 0;
// }
