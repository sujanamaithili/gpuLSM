#include <iostream>
#include <cuda_runtime.h>
#include "merge.cuh"

#include <vector>
#include <algorithm>
#include <chrono>

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
            d_merged[idx] = (d_arr1[start1].first < d_arr2[start2].first) ? d_arr1[start1] : d_arr2[start2];
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

// TODO: Remove after testing merge code

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
        std::cout << "(" << h_merged[i].first << ", " << h_merged[i].second << ") ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_arr1);
    cudaFree(d_arr2);
    cudaFree(d_merged);
}

// int main() {
//     testMerge();
//     return 0;
// }

// int main(int argc, char** argv) {
//     if (argc < 3) {
//         std::cerr << "Usage: " << argv[0] << " <size1> <size2>" << std::endl;
//         return 1;
//     }

//     int size1 = std::stoi(argv[1]);
//     int size2 = std::stoi(argv[2]);

//     // Generate random sorted arrays of Pair objects
//     std::vector<Pair<int, int>> arr1(size1), arr2(size2);
//     for (int i = 0; i < size1; ++i) arr1[i] = Pair<int, int>(rand() % 10000, rand() % 100);
//     for (int i = 0; i < size2; ++i) arr2[i] = Pair<int, int>(rand() % 10000, rand() % 100);
//     std::sort(arr1.begin(), arr1.end(), [](const Pair<int, int>& a, const Pair<int, int>& b) { return a.first < b.first; });
//     std::sort(arr2.begin(), arr2.end(), [](const Pair<int, int>& a, const Pair<int, int>& b) { return a.first < b.first; });

//     // Measure CPU time
//     auto cpu_start = std::chrono::high_resolution_clock::now();
//     std::vector<Pair<int, int>> mergedCPU = mergeCPU(arr1, arr2);
//     auto cpu_end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;
//     std::cout << "CPU merge time: " << cpu_duration.count() << " seconds" << std::endl;

//     // Copy data to GPU
//     Pair<int, int> *d_arr1, *d_arr2;
//     cudaMalloc(&d_arr1, size1 * sizeof(Pair<int, int>));
//     cudaMalloc(&d_arr2, size2 * sizeof(Pair<int, int>));


//     // Measure GPU time
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);

    
//     cudaMemcpy(d_arr1, arr1.data(), size1 * sizeof(Pair<int, int>), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_arr2, arr2.data(), size2 * sizeof(Pair<int, int>), cudaMemcpyHostToDevice);

    

//     Pair<int, int>* d_merged = merge(d_arr1, size1, d_arr2, size2);

//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     std::cout << "GPU merge time: " << milliseconds / 1000.0 << " seconds" << std::endl;

//     // Compare CPU and GPU results
//     bool isEqual = compareResults(mergedCPU, d_merged, size1 + size2);
//     std::cout << "Results match: " << (isEqual ? "Yes" : "No") << std::endl;

//     // Cleanup
//     cudaFree(d_arr1);
//     cudaFree(d_arr2);
//     cudaFree(d_merged);

//     return 0;
// }