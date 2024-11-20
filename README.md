# GPU-Accelerated Log-Structured Merge Tree Library

**GPU-LSM** is a high-performance library written in **C++** and **CUDA**, designed for **Log-Structured Merge (LSM) Trees**. It provides efficient implementations of key operations such as **insert**, **delete**, **lookup**, **count**, and **range queries**. The library leverages GPU acceleration for significant performance improvements over CPU-based counterparts.

---

## Folder Structure

- **`src/`**  
  - `lsm.cuh`: Class definition and kernel integration.
  - `main.cuh`: Basic tests for LSM tree functionality.
  
- **`kernels/`**  
  - CUDA kernels used in the library for sort, merge and some intermediate operations
   
- **`cpu/`**  
   - Provides CPU-only implementations of LSM operations for performance comparison.

---

## How to Use

### Building the Project

   ```bash
   git clone git@github.com:sujanamaithili/gpuLSM.git

   cd gpuLSM
  
   make
   ```

### Benchmarking
    
    ./benchmark.sh 

Output: 

```
[sb9509@cuda4 gpuLSM]$ ./benchmark.sh
nvcc -w -Ikernels -Isrc --expt-relaxed-constexpr -c src/main.cu -o src/main.o
nvcc -w -Ikernels -Isrc --expt-relaxed-constexpr src/main.o -o GPULSM
g++ -std=c++17 cpuTest.cpp -o cpuTest
Running GPU test for buffer size: 16
Running CPU test for buffer size: 16
Running GPU test for buffer size: 256
Running CPU test for buffer size: 256
Running GPU test for buffer size: 4096
Running CPU test for buffer size: 4096
Running GPU test for buffer size: 65536
Running CPU test for buffer size: 65536
Running GPU test for buffer size: 1048576
Running CPU test for buffer size: 1048576
Running GPU test for buffer size: 16777216
Running CPU test for buffer size: 16777216
+-------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+
| Buffer Size | GPU Init Time (s) | GPU Insert Time (s) | GPU Lookup Time (s) | CPU Init Time (s) | CPU Insert Time (s) | CPU Lookup Time (s) |
+-------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+
|          16 |          0.067102 |          0.000577 |          0.000056 |        1.1624e-05 |        3.4987e-05 |         4.669e-06 |
|         256 |          0.059488 |          0.001469 |          0.000062 |       0.000171562 |       0.000737305 |        8.5573e-05 |
|        4096 |          0.059954 |          0.003612 |          0.000085 |        0.00265831 |         0.0157912 |        0.00179668 |
|       65536 |          0.059532 |          0.015614 |          0.000843 |         0.0353684 |          0.311442 |         0.0359707 |
|     1048576 |          0.060606 |          0.182683 |          0.010905 |          0.557269 |           5.95755 |          0.707152 |
|    16777216 |          0.062982 |          3.138395 |          0.000017 |           8.92472 |           110.965 |           14.8201 |
+-------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+
```

### Library usage 

   This is a basic example demonstrated in src/example.cu

```cpp

 // This initializes an LSM tree with 3 levels, where the first level can accommodate 4 <float, char*> pairs.
 lsmTree<float, char*> tree(3, 4, false);

 Pair<float, char*> keyValuePairs[] = {{1.1f, "value1"}, {2.2f, "value2"}, {3.3f, "value3"}, {4.4f, "value4"}};

 // This inserts the key-value pairs. (Note: This library supports only batch inserts of size equal to level 0's capacity at once.)
 tree.updateKeys(keyValuePairs, bufferSize); 

 float queryKeys[] = {3.3f, 1.1f, 7.7f};
 char* results[3];
 bool foundFlags[3];

 // This queries the specified keys and stores the results in the `results` array. The `foundFlags` array is set to `true` for found keys. (Note: Lookups need not match the size of level 0.)
 tree.queryKeys(queryKeys, 3, results, foundFlags);

```


   
    
