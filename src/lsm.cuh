#ifndef GPU_LSM_TREE_H
#define GPU_LSM_TREE_H

template <typename Key, typename Value>
struct Pair {
    Key first;
    Value second;
    __host__ __device__ Pair() : first(Key()), second(Value()) {}
    __host__ __device__ Pair(const Key a, const Value b) : first(a), second(b) {}
};

template <typename Key, typename Value>
class lsmTree {
private:
    int numLevels; // No of levels in LSM
    int bufferSize; // Size of buffer
    int maxSize; // Max Size of LSM tree
    int numBatches; // No of batches pushed each of buffer size
    Pair<Key, Value>* memory; // Array of key value pairs for all levels

    __host__ void incrementBatchCounter() { numBatches++; }
    __host__ int getNumBatches() const { return numBatches; }
    __host__ Pair<Key, Value>* getMemory() const { return memory; }

public:
    __host__ __device__ lsmTree(int numLevels, int bufferSize);

    __host__ __device__ ~lsmTree();

    __host__ bool updateKeys(const Pair<Key, Value>* kv, int size);

    __device__ bool deleteKeys(const Key* keys, int size);

    __host__ void queryKeys(const Key* keys, int size, Value* results, bool* foundFlags);
};

#endif // GPU_LSM_TREE_H
