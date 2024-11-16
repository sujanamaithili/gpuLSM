#ifndef GPU_LSM_TREE_H
#define GPU_LSM_TREE_H

template <typename T>
class Sentinel {
public:
    static T& tombstone() {
        static T instance = T(); // Singleton instance acting as the sentinel
        return instance;
    }
};

template <typename Key, typename Value>
struct Pair {
    Key first;
    Value second;
    __host__ __device__ Pair() : first(Key()), second(Value()) {}
    __host__ __device__ Pair(const Key a, const Value b) : first(a), second(b) {}
    __host__ __device__ void setTombstone() { second = Sentinel<Value>::tombstone(); }
    __host__ __device__ bool isTombstone() const { return &second == &Sentinel<Value>::tombstone();}
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

    __host__ __device__ lsmTree(int numLevels, int bufferSize);

    __host__ __device__ ~lsmTree();

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
