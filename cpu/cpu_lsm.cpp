#include "cpu_lsm.h"
#include <iostream>
#include <unordered_map>
#include <iterator>
#include <cstdlib>

using namespace std;

template <typename Key, typename Value>
lsmTree<Key, Value>::lsmTree(int numLevels, int bufferSize)
    : numLevels(numLevels), bufferSize(bufferSize), numBatches(0) {
    this->maxSize = 0;
    for (int i = 0; i < numLevels; i++) {
        this->maxSize += bufferSize * (1 << i);
    }
    memory = new Pair<Key, Value>[maxSize];
}

template <typename Key, typename Value>
lsmTree<Key, Value>::~lsmTree() {
    delete[] memory;
}

template <typename Key, typename Value>
bool lsmTree<Key, Value>::updateKeys(const std::vector<Pair<Key, Value>>& kv) {
    for (size_t i = 0; i < kv.size(); i++) {
        memory[numBatches * bufferSize + i] = kv[i];
    }
    incrementBatchCounter();
    return true;
}
