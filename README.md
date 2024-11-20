#GPU-Accelerated Log-Structured Merge Tree Library

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


   
    
