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
void runTestWithDeletedKeys() {
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
        kvPairs[i] = {i + 1, (i + 1) * 10};
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

    // Define keys to delete
    const int numKeysToDelete = 32;
    Key keysToDelete[numKeysToDelete];
    for (int i = 0; i < numKeysToDelete; ++i) {
        keysToDelete[i] = (i + 1) * 2; // Delete even keys [2, 4, 6, ..., 32]
    }

    // Delete the specified keys
    if (!tree.deleteKeys(keysToDelete, numKeysToDelete)) {
        printf("Error: Deletion failed for keys.\n");
        return;
    }
    printf("Successfully deleted %d keys.\n", numKeysToDelete);
    tree.printAllLevels();
    // Query the LSM tree to check the deleted keys
    printf("\nTesting correctness of deleted key queries:\n");
    int numCorrect = 0;
    bool foundFlags[numKeysToDelete];
    Value results[numKeysToDelete];

    tree.queryKeys(keysToDelete, numKeysToDelete, results, foundFlags);

    // Verify that deleted keys return a tombstone value
    for (int i = 0; i < numKeysToDelete; ++i) {
        if (!foundFlags[i] || results[i] == Sentinel<Value>::tombstone()) {
            numCorrect++;
        } else {
            printf("Error: Key %d expected tombstone but got value %d.\n", keysToDelete[i], results[i]);
        }
    }

    printf("\nCorrectly handled %d out of %d deleted keys.\n", numCorrect, numKeysToDelete);
    if (numCorrect == numKeysToDelete) {
        printf("Test passed: All deleted keys were correctly identified as tombstoned.\n");
    } else {
        printf("Test failed: Some deleted keys were not correctly identified as tombstoned.\n");
    }

    // Verify that non-deleted keys still return correct values
    printf("\nTesting correctness of non-deleted key queries:\n");
    numCorrect = 0;
    const int numNonDeletedKeys = 32;
    Key nonDeletedKeys[numNonDeletedKeys];
    Value expectedValues[numNonDeletedKeys];
    bool foundFlagsNonDeleted[numNonDeletedKeys];
    Value resultsNonDeleted[numNonDeletedKeys];

    for (int i = 0; i < numNonDeletedKeys; ++i) {
        nonDeletedKeys[i] = (i * 2) + 1; // Query odd keys [1, 3, 5, ..., 31]
        expectedValues[i] = nonDeletedKeys[i] * 10;
    }

    tree.queryKeys(nonDeletedKeys, numNonDeletedKeys, resultsNonDeleted, foundFlagsNonDeleted);

    // Verify results for non-deleted keys
    for (int i = 0; i < numNonDeletedKeys; ++i) {
        if (foundFlagsNonDeleted[i] && resultsNonDeleted[i] == expectedValues[i]) {
            numCorrect++;
        } else {
            printf("Error: Key %d expected value %d but got %d.\n", nonDeletedKeys[i], expectedValues[i], resultsNonDeleted[i]);
        }
    }

    printf("\nCorrectly retrieved %d out of %d non-deleted key-value pairs.\n", numCorrect, numNonDeletedKeys);
    if (numCorrect == numNonDeletedKeys) {
        printf("Test passed: All non-deleted key-value pairs were correctly retrieved.\n");
    } else {
        printf("Test failed: Some non-deleted key-value pairs were not correctly retrieved.\n");
    }
}

int main() {
    printf("Running Test with Unique Keys:\n");
    runTestWithUniqueKeys();

    printf("\nRunning Test with Duplicate Keys:\n");
    runTestWithDuplicateKeys();

    printf("\nRunning Test with Deleted Keys:\n");
    runTestWithDeletedKeys();

    return 0;
}

