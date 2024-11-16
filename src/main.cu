#include <bits/stdc++.h>
#include <lsm.cuh>

void runTestWithUniqueKeys() {
    // Define the data types for the key and value
    using Key = int;
    using Value = int;

    // Parameters for the LSM tree
    const int numLevels = 4;
    const int bufferSize = 32;

    // Create an instance of the LSM tree
    lsmTree<Key, Value> tree(numLevels, bufferSize);

    // Define 96 key-value pairs for insertion
    const int totalPairs = 96;
    Pair<Key, Value> kvPairs[totalPairs];
    for (int i = 0; i < totalPairs; ++i) {
        kvPairs[i] = {totalPairs - (i + 1), (totalPairs - (i + 1)) * 10};
    }

    // Randomize the key-value pairs
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(std::begin(kvPairs), std::end(kvPairs), gen);

    // Insert in batches of 32 key-value pairs
    const int batchSize = 32;
    for (int i = 0; i < totalPairs; i += batchSize) {
        if (!tree.updateKeys(kvPairs + i, batchSize)) {
            printf("Error: Insertion failed for batch starting at index %d.\n", i);
            return;
        }
        printf("Successfully inserted batch starting at index %d.\n", i);
    }

    // Query the LSM tree
    printf("\nTesting correctness of unique key-value pairs:\n");
    int numCorrect = 0;
    bool foundFlags[totalPairs];
    Value results[totalPairs];
    Key keysToQuery[totalPairs];

    for (int i = 0; i < totalPairs; ++i) {
        keysToQuery[i] = kvPairs[i].first;
    }

    tree.queryKeys(keysToQuery, totalPairs, results, foundFlags);

    // Verify results
    for (int i = 0; i < totalPairs; ++i) {
        if (foundFlags[i] && results[i] == kvPairs[i].second) {
            numCorrect++;
        } else {
            printf("Error: Key %d expected value %d but got %d.\n", keysToQuery[i], kvPairs[i].second, results[i]);
        }
    }

    printf("\nCorrectly retrieved %d out of %d key-value pairs.\n", numCorrect, totalPairs);
    if (numCorrect == totalPairs) {
        printf("Test passed: All key-value pairs were correctly inserted and retrieved.\n");
    } else {
        printf("Test failed: Some key-value pairs were not correctly inserted or retrieved.\n");
    }
}

void runTestWithDuplicateKeys() {
    // Define the data types for the key and value
    using Key = int;
    using Value = int;

    // Parameters for the LSM tree
    const int numLevels = 4;
    const int bufferSize = 32;

    // Create an instance of the LSM tree
    lsmTree<Key, Value> tree(numLevels, bufferSize);

    // Define key-value pairs with duplicate keys
    const int totalPairs = 96;
    Pair<Key, Value> kvPairs[totalPairs];
    for (int i = 0; i < totalPairs; ++i) {
        int key = (i % 32) + 1; // Duplicate keys in the range [1, 32]
        kvPairs[i] = {key, (i + 1) * 100}; // New value for each duplicate key
    }

    // Insert in batches of 32 key-value pairs
    const int batchSize = 32;
    for (int i = 0; i < totalPairs; i += batchSize) {
        if (!tree.updateKeys(kvPairs + i, batchSize)) {
            printf("Error: Insertion failed for batch starting at index %d.\n", i);
            return;
        }
        printf("Successfully inserted batch starting at index %d.\n", i);
    }

    // Query the LSM tree
    printf("\nTesting correctness with duplicate key-value pairs:\n");
    int numCorrect = 0;
    bool foundFlags[32];
    Value results[32];
    Key keysToQuery[32];

    // Query only for the keys in the range [1, 32]
    for (int i = 0; i < 32; ++i) {
        keysToQuery[i] = i + 1;
    }

    tree.queryKeys(keysToQuery, 32, results, foundFlags);

    // Verify that each key returns the last inserted value
    for (int i = 0; i < 32; ++i) {
        Value expectedValue = (i + 64 + 1) * 100; // Last inserted value for each key
        if (foundFlags[i] && results[i] == expectedValue) {
            numCorrect++;
        } else {
            printf("Error: Key %d expected value %d but got %d.\n", keysToQuery[i], expectedValue, results[i]);
        }
    }

    printf("\nCorrectly retrieved %d out of 32 key-value pairs with duplicate keys.\n", numCorrect);
    if (numCorrect == 32) {
        printf("Test passed: All duplicate key-value pairs returned the last inserted value.\n");
    } else {
        printf("Test failed: Some duplicate key-value pairs did not return the last inserted value.\n");
    }
}

int main() {
    printf("Running Test with Unique Keys:\n");
    runTestWithUniqueKeys();

    printf("\nRunning Test with Duplicate Keys:\n");
    runTestWithDuplicateKeys();

    return 0;
}
