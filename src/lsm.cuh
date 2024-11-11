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

public:
    __host__ __device__ lsmTree(int numLevels, int bufferSize);

    __host__ __device__ ~lsmTree();

    __device__ bool updateKeys(const Key* keys, const Value* values, int size);

    __device__ bool deleteKeys(const Key* keys, int size);

    __device__ Value* queryKeys(const Key* keys, Value* results, int size);
};

#endif // GPU_LSM_TREE_H
