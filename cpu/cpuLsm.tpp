#include "cpuLsm.h"
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

template <typename Key, typename Value>
void lsmTree<Key, Value>::queryKeys(const std::vector<Key>& keys, std::vector<Value>& results, std::vector<bool>& foundFlags) {
    results.resize(keys.size());
    foundFlags.resize(keys.size());

    for (size_t i = 0; i < keys.size(); ++i) {
        Key key = keys[i];
        bool found = false;

        for (int level = 0; level < numLevels; ++level) {

            // Skip this level if it hasn't been filled
            if (!(numBatches & (1 << level))) {
                continue;
            }

            int levelStart = (1 << level) - 1;  // Start index of this level
            int levelSize = bufferSize * (1 << level);  // Size of this level
            int levelEnd = levelStart + levelSize;  // End index of this level

            // Perform binary search for the key in the current level
            int left = levelStart;
            int right = levelEnd - 1;
            int firstOccurrence = -1;

            while (left <= right) {
                int mid = left + (right - left) / 2;

                if (memory[mid].isKeyEmpty()) {
                    // Skip empty slots, adjust search range
                    right = mid - 1;
                    continue;
                }

                if (memory[mid].first.has_value()) {
                    Key midKey = memory[mid].first.value();
                    if (midKey == key) {
                        // Key found, record first occurrence and continue searching left
                        firstOccurrence = mid;
                        right = mid - 1;
                    } else if (midKey < key) {
                        left = mid + 1;
                    } else {
                        right = mid - 1;
                    }
                } else {
                    // Invalid or empty key
                    right = mid - 1;
                }
            }

            if (firstOccurrence != -1) {
                // Key found at the first occurrence
                found = true;
                foundFlags[i] = true;
                if (!memory[firstOccurrence].isValueTombstone()) {
                    results[i] = memory[firstOccurrence].second.value();
                } else {
                    // Tombstone
                    results[i] = Value();
                    foundFlags[i] = false;
                }
                break;
            }
        }

        if (!found) {
            results[i] = Value(); // Default value if not found
            foundFlags[i] = false;
        }
    }
}


template <typename Key, typename Value>
void lsmTree<Key, Value>::countKeys(const std::vector<Key>& k1, const std::vector<Key>& k2, int numQueries, std::vector<int>& counts) {
    // Initialize counts for each query
    counts.resize(numQueries, 0);

    std::vector<int> l(numQueries * numLevels, 0);
    std::vector<int> u(numQueries * numLevels, 0);
    std::vector<int> init_count(numQueries * numLevels, 0);

    for (int queryIdx = 0; queryIdx < numQueries; ++queryIdx) {
        Key lowerBound = k1[queryIdx];
        Key upperBound = k2[queryIdx];

        for (int level = 0; level < numLevels; ++level) {
            int levelStart = (1 << level) - 1;
            int levelSize = (1 << level) * bufferSize;

            int idx = queryIdx * numLevels + level;

            l[idx] = std::distance(memory + levelStart, std::lower_bound(memory + levelStart, memory + levelStart + levelSize, Pair<Key, Value>(lowerBound, {}), [](const Pair<Key, Value>& a, const Pair<Key, Value>& b) {
                return a.first < b.first;
            }));
            u[idx] = std::distance(memory + levelStart, std::upper_bound(memory + levelStart, memory + levelStart + levelSize, Pair<Key, Value>(upperBound, {}), [](const Pair<Key, Value>& a, const Pair<Key, Value>& b) {
                return a.first < b.first;
            }));

            init_count[idx] = u[idx] - l[idx];
        }

        std::vector<int> offset(numQueries * numLevels, 0);
        for (int queryIdx = 0; queryIdx < numQueries; ++queryIdx) {
            int runningSum = 0;
            for (int level = 0; level < numLevels; ++level) {
                int idx = queryIdx * numLevels + level;
                offset[idx] = runningSum;
                runningSum += init_count[idx];
            }
        }

        std::vector<std::vector<Pair<Key, Value>>> results(numQueries);
        for (int queryIdx = 0; queryIdx < numQueries; ++queryIdx) {
            for (int level = 0; level < numLevels; ++level) {
                int idx = queryIdx * numLevels + level;
                int start = l[idx];
                int end = u[idx];
                int levelStart = (1 << level) - 1;

                for (int i = start; i < end; ++i) {
                    results[queryIdx].push_back(memory[levelStart + i]);
                }
            }
        }

        for (int queryIdx = 0; queryIdx < numQueries; ++queryIdx) {
            std::stable_sort(results[queryIdx].begin(), results[queryIdx].end(), [](const Pair<Key, Value>& a, const Pair<Key, Value>& b) {
                return a.first < b.first;
            });

            // Count unique keys (skip tombstones and duplicates)
            std::optional<Key> lastKey;
            bool firstKey = true;
            counts[queryIdx] = 0;

            for (auto pair : results[queryIdx]) {
                if (firstKey || pair.first != lastKey) {
                    if (!pair.isValueTombstone()) { 
                        counts[queryIdx]++;
                    }
                    lastKey = pair.first;
                    firstKey = false;
                }
            }
        }

    }
}


template <typename Key, typename Value>
void lsmTree<Key, Value>::rangeKeys(const std::vector<Key>& k1, const std::vector<Key>& k2, int numQueries, std::vector<std::vector<Pair<Key, Value>>>& results) {
    std::vector<int> l(numQueries * numLevels, 0);
    std::vector<int> u(numQueries * numLevels, 0);
    std::vector<int> init_count(numQueries * numLevels, 0);

    for (int queryIdx = 0; queryIdx < numQueries; ++queryIdx) {
        Key lowerBound = k1[queryIdx];
        Key upperBound = k2[queryIdx];

        for (int level = 0; level < numLevels; ++level) {
            int levelStart = (1 << level) - 1;
            int levelSize = (1 << level) * bufferSize;

            int idx = queryIdx * numLevels + level;

            l[idx] = std::distance(memory + levelStart, std::lower_bound(memory + levelStart, memory + levelStart + levelSize, Pair<Key, Value>(lowerBound, {}), [](const Pair<Key, Value>& a, const Pair<Key, Value>& b) {
                return a.first < b.first;
            }));
            u[idx] = std::distance(memory + levelStart, std::upper_bound(memory + levelStart, memory + levelStart + levelSize, Pair<Key, Value>(upperBound, {}), [](const Pair<Key, Value>& a, const Pair<Key, Value>& b) {
                return a.first < b.first;
            }));

            init_count[idx] = u[idx] - l[idx];
        }

        std::vector<int> offset(numQueries * numLevels, 0);
        for (int queryIdx = 0; queryIdx < numQueries; ++queryIdx) {
            int runningSum = 0;
            for (int level = 0; level < numLevels; ++level) {
                int idx = queryIdx * numLevels + level;
                offset[idx] = runningSum;
                runningSum += init_count[idx];
            }
        }

        std::vector<std::vector<Pair<Key, Value>>> results(numQueries);
        for (int queryIdx = 0; queryIdx < numQueries; ++queryIdx) {
            for (int level = 0; level < numLevels; ++level) {
                int idx = queryIdx * numLevels + level;
                int start = l[idx];
                int end = u[idx];
                int levelStart = (1 << level) - 1;

                for (int i = start; i < end; ++i) {
                    results[queryIdx].push_back(memory[levelStart + i]);
                }
            }
        }

        for (int queryIdx = 0; queryIdx < numQueries; ++queryIdx) {
            std::stable_sort(results[queryIdx].begin(), results[queryIdx].end(), [](const Pair<Key, Value>& a, const Pair<Key, Value>& b) {
                return a.first < b.first;
            });

            // Count unique keys (skip tombstones and duplicates)
            std::vector<Pair<Key, Value>> compacted;
            std::optional<Key> lastKey;
            bool firstKey = true;

            for (auto pair : results[queryIdx]) {
                if (firstKey || pair.first != lastKey) {
                    if (!pair.isValueTombstone()) { 
                        compacted.push_back(pair);
                    }
                    lastKey = pair.first;
                    firstKey = false;
                }
            }
            results[queryIdx] = std::move(compacted);
        }

    }
}

