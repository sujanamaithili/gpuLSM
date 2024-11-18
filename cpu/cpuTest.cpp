#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include "cpuLsm.h"

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

int main() {
    testUpdateKeys();
    testQueryKeys();
    testCountKeys();
    testRangeKeys();

    std::cout << "All tests passed successfully.\n";
    return 0;
}