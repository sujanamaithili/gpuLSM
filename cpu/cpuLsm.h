#ifndef CPU_LSM_H
#define CPU_LSM_H

#include <vector>
#include <optional>
#include <algorithm>
#include <cstdio>


template <typename Key, typename Value>
struct Pair {
    std::optional<Key> first;
    std::optional<Value> second;

    Pair() : first(std::nullopt), second(std::nullopt) {}
    Pair(const std::optional<Key>& a, const std::optional<Value>& b) : first(a), second(b) {}
    Pair(std::nullopt_t, std::nullopt_t) : first(std::nullopt), second(std::nullopt) {}

    void setKeyEmpty() { first = std::nullopt; }
    void setValueTombstone() { second = std::nullopt; }

    bool isKeyEmpty() const { return !first.has_value(); }
    bool isValueTombstone() const { return !second.has_value(); }

    // Define operator== for comparison
    bool operator==(const Pair& other) const {
        return first == other.first && second == other.second;
    }
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

    lsmTree(int numLevels, int bufferSize);

    ~lsmTree();

    bool updateKeys(const std::vector<Pair<Key, Value>>& kv);

    bool deleteKeys(const std::vector<Key>& keys);

    void queryKeys(const std::vector<Key>& keys, std::vector<Value>& results, std::vector<bool>& foundFlags);

    void countKeys(const std::vector<Key>& k1, const std::vector<Key>& k2, int numQueries, std::vector<int>& counts);
    
    void rangeKeys(const std::vector<Key>& k1, const std::vector<Key>& k2, int numQueries, std::vector<std::vector<Pair<Key, Value>>>& results);

    void printLevel(int level) const;

    void printAllLevels() const;

    void incrementBatchCounter() { numBatches++; }

    int getNumBatches() const { return numBatches; }
    
    Pair<Key, Value>* getMemory() const { return memory; }

    int getNumLevels() const { return numLevels; }
    
    int getBufferSize() const { return bufferSize; }

};

#include "cpuLsm.tpp"

#endif