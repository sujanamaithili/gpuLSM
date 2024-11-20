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
|          16 |          0.114158 |          0.000571 |          0.000063 |        1.2515e-05 |        3.7787e-05 |         6.141e-06 |
|         256 |          0.109488 |          0.001467 |          0.000059 |       0.000170414 |         0.0007428 |        8.6594e-05 |
|        4096 |          0.124131 |          0.003615 |          0.000087 |        0.00263562 |         0.0156988 |        0.00179585 |
|       65536 |          0.079597 |          0.015640 |          0.000880 |         0.0353149 |          0.311165 |         0.0359587 |
|     1048576 |          0.114220 |          0.187476 |          0.012942 |          0.557599 |             5.969 |          0.716547 |
|    16777216 |          0.139053 |          3.306873 |          0.593892 |           8.92855 |           111.071 |           14.9114 |
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


   
    
