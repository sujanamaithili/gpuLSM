#include <bits/stdc++.h>
#include <lsm.cuh>

int main() {
    // Define the data types for the key and value
    using Key = int;
    using Value = int;

    // Parameters for the LSM tree
    const int numLevels = 4;      // Number of levels
    const int bufferSize = 32;    // Buffer size (initial size of level 0)

    // Create an instance of the LSM tree
    lsmTree<Key, Value> tree(numLevels, bufferSize);

    // Define 96 key-value pairs for insertion
    const int totalPairs = 96;
    Pair<Key, Value> kvPairs[totalPairs];
    for (int i = 0; i < totalPairs; ++i) {
        kvPairs[i] = {totalPairs - (i + 1), (totalPairs - (i + 1)) * 10};
    }

    // Randomize the key-value pairs
    std::random_device rd;  // Seed for random number generation
    std::mt19937 gen(rd()); // Mersenne Twister random number generator
    std::shuffle(std::begin(kvPairs), std::end(kvPairs), gen);

    // Insert in batches of 32 key-value pairs
    const int batchSize = 32;
    for (int i = 0; i < totalPairs; i += batchSize) {
        if (!tree.updateKeys(kvPairs + i, batchSize)) {
            printf("Error: Insertion failed for batch starting at index %d.\n", i);
            return 1;
        }
        printf("Successfully inserted batch starting at index %d.\n", i);
    }

    // Print the LSM tree structure after the insertion
    printf("\nLSM Tree Structure after inserting 96 key-value pairs in batches of 32:\n");
    tree.printAllLevels();

    // Test correctness by querying all inserted keys
    printf("\nTesting correctness of inserted key-value pairs:\n");
    int numCorrect = 0;
    bool foundFlags[totalPairs];
    Value results[totalPairs];

    // Extract keys for querying
    Key keysToQuery[totalPairs];
    for (int i = 0; i < totalPairs; ++i) {
        keysToQuery[i] = kvPairs[i].first;
    }

    // Query the LSM tree
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

    return 0;
}

