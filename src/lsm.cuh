#ifndef GPU_LSM_TREE_H
#define GPU_LSM_TREE_H

#include <optional>

template <typename Key, typename Value>
struct Pair {
    std::optional<Key> first;
    std::optional<Value> second;

    __host__ __device__ Pair() : first(std::nullopt), second(std::nullopt) {}
    __host__ __device__ Pair(const std::optional<Key>& a, const std::optional<Value>& b) : first(a), second(b) {}
    __host__ __device__ Pair(std::nullopt_t, std::nullopt_t) : first(std::nullopt), second(std::nullopt) {}

    __host__ __device__ void setKeyEmpty() { first = std::nullopt; }
    __host__ __device__ void setValueTombstone() { second = std::nullopt; }

    __host__ __device__ bool isKeyEmpty() const { return !first.has_value(); }
    __host__ __device__ bool isValueTombstone() const { return !second.has_value(); }
};

template <typename Key, typename Value>
class lsmTree {
private:
    int numLevels; // No of levels in LSM
    int bufferSize; // Size of buffer
    int maxSize; // Max Size of LSM tree
    int numBatches; // No of batches pushed each of buffer size

public:
    Pair<Key, Value>* memory; // Array of key value pairs for all levels

    __host__  lsmTree(int numLevels, int bufferSize);

    __host__  ~lsmTree();

    __host__ bool updateKeys(const Pair<Key, Value>* kv, int size);

    __host__ bool deleteKeys(const Key* keys, int size);

    __host__ void queryKeys(const Key* keys, int size, Value* results, bool* foundFlags);

    __host__ __device__ void printLevel(int level) const;

    __host__ __device__ void printAllLevels() const;

    __host__ void countKeys(const Key* k1, const Key* k2, int numQueries, int* counts);

    __host__ __device__ void incrementBatchCounter() { numBatches++; }
    __host__ __device__ int getNumBatches() const { return numBatches; }
    __host__ __device__ Pair<Key, Value>* getMemory() const { return memory; }

    __host__ __device__ int getNumLevels() const { return numLevels; }
    __host__ __device__ int getBufferSize() const { return bufferSize; }

};

#endif // GPU_LSM_TREE_H

