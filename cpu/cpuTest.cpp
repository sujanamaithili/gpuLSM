#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include "cpuLsm.h"
#include <bits/stdc++.h>

void testUpdateKeys() {
    lsmTree<int, std::string> tree(3, 4);

    std::vector<Pair<int, std::string>> data = {
        {1, "value1"}, {2, "value2"}, {3, "value3"}, {4, "value4"}
    };

    bool result = tree.updateKeys(data);
    assert((result == true));

    std::vector<std::string> results;
    std::vector<bool> foundFlags;
    std::vector<int> keys = {1, 2, 3, 4};
    tree.queryKeys(keys, results, foundFlags);

    // Expected results
    std::vector<std::string> expectedResults = {"value1", "value2", "value3", "value4"};
    std::vector<bool> expectedFoundFlags = {true, true, true, true};

    // Check results with for loop
    for (size_t i = 0; i < results.size(); ++i) {
        assert(results[i] == expectedResults[i]);
    }

    // Check found flags with for loop
    for (size_t i = 0; i < foundFlags.size(); ++i) {
        assert(foundFlags[i] == expectedFoundFlags[i]);
    }

    std::cout << "testUpdateKeys passed.\n";
}

void testQueryKeys() {
    lsmTree<int, std::string> tree(3, 4);

    std::vector<Pair<int, std::string>> data = {
        {10, "ten"}, {20, "twenty"}, {30, "thirty"}
    };
    tree.updateKeys(data);

    std::vector<std::string> results;
    std::vector<bool> foundFlags;
    std::vector<int> keys = {10, 20, 30, 40};
    tree.queryKeys(keys, results, foundFlags);

    // Expected results
    std::vector<std::string> expectedResults = {"ten", "twenty", "thirty", ""};
    std::vector<bool> expectedFoundFlags = {true, true, true, false};

    // Check results with for loop
    for (size_t i = 0; i < results.size(); ++i) {
        assert(results[i] == expectedResults[i]);
    }

    // Check found flags with for loop
    for (size_t i = 0; i < foundFlags.size(); ++i) {
        assert(foundFlags[i] == expectedFoundFlags[i]);
    }

    std::cout << "testQueryKeys passed.\n";
}

void testCountKeys() {
    lsmTree<int, std::string> tree(3, 4);

    std::vector<Pair<int, std::string>> data = {
        {5, "five"}, {15, "fifteen"}, {25, "twenty-five"}
    };
    tree.updateKeys(data);

    std::vector<int> k1 = {0, 10};
    std::vector<int> k2 = {10, 30};
    std::vector<int> counts;
    tree.countKeys(k1, k2, 2, counts);

    // Expected counts
    std::vector<int> expectedCounts = {1, 2};

    // Check counts with for loop
    for (size_t i = 0; i < counts.size(); ++i) {
        assert(counts[i] == expectedCounts[i]);
    }

    std::cout << "testCountKeys passed.\n";
}

void testRangeKeys() {
    lsmTree<int, std::string> tree(3, 4);

    std::vector<Pair<int, std::string>> data = {
        {1, "one"}, {2, "two"}, {3, "three"}, {4, "four"}
    };
    tree.updateKeys(data);

    std::vector<int> k1 = {1};
    std::vector<int> k2 = {3};
    std::vector<std::vector<Pair<int, std::string>>> results;
    tree.rangeKeys(k1, k2, 1, results);

    // Expected results
    std::vector<Pair<int, std::string>> expected = {
        {1, "one"}, {2, "two"}, {3, "three"}
    };

    // Check range keys with for loop
    for (size_t i = 0; i < results[0].size(); ++i) {
        assert(results[0][i] == expected[i]);
    }

    std::cout << "testRangeKeys passed.\n";
}

void testLSMTree(const std::vector<int>& testSizes) {
    using Key = int;
    using Value = int;

    for (int numUpdates : testSizes) {
        if (numUpdates <= 0 || (numUpdates & (numUpdates - 1)) != 0) {
            std::cout << "Error: Invalid size " << numUpdates << ". Skipping.\n";
            continue;
        }

        const int numLevels = 4;
        const int bufferSize = numUpdates/4;

        // Measure initialization time
        auto initStart = std::chrono::high_resolution_clock::now();
        lsmTree<Key, Value> tree(numLevels, bufferSize);
        auto initEnd = std::chrono::high_resolution_clock::now();
        std::cout << "Initialized LSM tree with " << numUpdates 
                  << " keys in " 
                  << std::chrono::duration<double>(initEnd - initStart).count() 
                  << " seconds.\n";

        // Generate random key-value pairs
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<Key> keyDist(1, 1000);
        std::uniform_int_distribution<Value> valueDist(1, 10000);

        std::vector<Pair<Key, Value>> kvPairs(numUpdates);
        for (int i = 0; i < numUpdates; ++i) {
            kvPairs[i] = Pair<Key, Value>(
                std::make_optional(keyDist(gen)),
                std::make_optional(valueDist(gen))
            );
        }

        // Measure update time
        auto updateStart = std::chrono::high_resolution_clock::now();
        if (!tree.updateKeys(kvPairs)) {
            std::cout << "Error: Update failed for size " << numUpdates << ".\n";
            continue;
        }
        auto updateEnd = std::chrono::high_resolution_clock::now();
        std::cout << "Updated " << numUpdates 
                  << " keys in " 
                  << std::chrono::duration<double>(updateEnd - updateStart).count() 
                  << " seconds.\n";
    }
}

int main() {
    // testUpdateKeys();
    // testQueryKeys();
    // testCountKeys();
    // testRangeKeys();

    std::vector<int> testSizes = {16, 256, 4096, 65536, 1048576, 16777216};
    testLSMTree(testSizes);

    // std::cout << "All tests passed successfully.\n";
    return 0;
}

