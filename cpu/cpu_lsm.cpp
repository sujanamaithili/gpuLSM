#include "cpu_lsm.h"
#include <iostream>
#include <unordered_map>
#include <iterator>
#include <cstdlib>
#include <algorithm>
#include <vector> 

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
    // Step 1: Sort the incoming buffer (stable sort to preserve order)
    std::vector<Pair<Key, Value>> tempBuffer = kv;
    std::stable_sort(tempBuffer.begin(), tempBuffer.end(), [](const Pair<Key, Value>& a, const Pair<Key, Value>& b) {
        return a.first < b.first; 
    });

    // Step 2: Try to push the buffer into levels, starting from level 0
    for (int level = 0; level < numLevels; ++level) {
        int levelStart = (1 << level) - 1; 
        int levelSize = (1 << level) * bufferSize; 

        // Create a vector for the current level's data
        std::vector<Pair<Key, Value>> currentLevel;
        for (int i = levelStart; i < levelStart + levelSize; ++i) {
            if (!memory[i].isKeyEmpty()) {
                currentLevel.push_back(memory[i]);
            }
        }

        // Merge tempBuffer and currentLevel if the current level is full
        if (currentLevel.size() == levelSize) {
            std::vector<Pair<Key, Value>> mergedBuffer;
            mergedBuffer.reserve(tempBuffer.size() + currentLevel.size());

            // Merge the sorted buffers (stable merge)
            std::merge(tempBuffer.begin(), tempBuffer.end(),
                       currentLevel.begin(), currentLevel.end(),
                       std::back_inserter(mergedBuffer), [](const Pair<Key, Value>& a, const Pair<Key, Value>& b) {
                           return a.first < b.first;
                       });

            // Update tempBuffer with the merged result for the next level
            tempBuffer = mergedBuffer;

            // Clear the current level
            for (int i = levelStart; i < levelStart + levelSize; ++i) {
                memory[i].setKeyEmpty();
                memory[i].setValueTombstone();
            }
        } 
        else {
            // Found empty level, insert tempBuffer into lsm
            int bufferIndex = 0; 
            for (int i = levelStart; i < levelStart + levelSize; i++) {
                memory[i] = tempBuffer[bufferIndex++];
            }
            incrementBatchCounter();
            return true;
        }
    }

    // If we've exhausted all levels and couldn't insert
    fprintf(stderr, "LSM Tree is full. Cannot insert keys.\n");
    return false;
}

