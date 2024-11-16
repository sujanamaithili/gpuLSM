#include <bits/stdc++.h>
#include <lsm.cuh>

void runTestWithUniqueKeys() {
    using Key = int;
    using Value = int;
    const int numLevels = 4;
    const int bufferSize = 32;

    lsmTree<Key, Value> tree(numLevels, bufferSize);

    const int totalPairs = 96;
    Pair<Key, Value> kvPairs[totalPairs];
    for (int i = 0; i < totalPairs; ++i) {
        kvPairs[i] = Pair<Key, Value>(
            std::make_optional(totalPairs - (i + 1)),
            std::make_optional((totalPairs - (i + 1)) * 10)
        );
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(std::begin(kvPairs), std::end(kvPairs), gen);

    const int batchSize = 32;
    for (int i = 0; i < totalPairs; i += batchSize) {
        if (!tree.updateKeys(kvPairs + i, batchSize)) {
            printf("Error: Insertion failed for batch starting at index %d.\n", i);
            return;
        }
        printf("Successfully inserted batch starting at index %d.\n", i);
    }

    printf("\nTesting correctness of unique key-value pairs:\n");
    int numCorrect = 0;
    bool foundFlags[totalPairs];
    Value results[totalPairs];
    Key keysToQuery[totalPairs];

    for (int i = 0; i < totalPairs; ++i) {
        keysToQuery[i] = *kvPairs[i].first;
    }

    tree.queryKeys(keysToQuery, totalPairs, results, foundFlags);

    for (int i = 0; i < totalPairs; ++i) {
        if (foundFlags[i] && results[i] == *kvPairs[i].second) {
            numCorrect++;
        } else {
            printf("Error: Key %d expected value %d but got %d.\n", 
                   keysToQuery[i], 
                   *kvPairs[i].second, 
                   results[i]);
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
    using Key = int;
    using Value = int;
    const int numLevels = 4;
    const int bufferSize = 32;

    lsmTree<Key, Value> tree(numLevels, bufferSize);

    const int totalPairs = 96;
    Pair<Key, Value> kvPairs[totalPairs];
    for (int i = 0; i < totalPairs; ++i) {
        int key = (i % 32) + 1;
        kvPairs[i] = Pair<Key, Value>(
            std::make_optional(key),
            std::make_optional((i + 1) * 100)
        );
    }

    const int batchSize = 32;
    for (int i = 0; i < totalPairs; i += batchSize) {
        if (!tree.updateKeys(kvPairs + i, batchSize)) {
            printf("Error: Insertion failed for batch starting at index %d.\n", i);
            return;
        }
        printf("Successfully inserted batch starting at index %d.\n", i);
    }

    printf("\nTesting correctness with duplicate key-value pairs:\n");
    int numCorrect = 0;
    bool foundFlags[32];
    Value results[32];
    Key keysToQuery[32];

    for (int i = 0; i < 32; ++i) {
        keysToQuery[i] = i + 1;
    }

    tree.queryKeys(keysToQuery, 32, results, foundFlags);

    for (int i = 0; i < 32; ++i) {
        Value expectedValue = (i + 64 + 1) * 100;
        if (foundFlags[i] && results[i] == expectedValue) {
            numCorrect++;
        } else {
            printf("Error: Key %d expected value %d but got %d.\n", 
                   keysToQuery[i], 
                   expectedValue, 
                   results[i]);
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
    using Key = int;
    using Value = int;
    const int numLevels = 4;
    const int bufferSize = 32;

    lsmTree<Key, Value> tree(numLevels, bufferSize);

    const int totalPairs = 96;
    Pair<Key, Value> kvPairs[totalPairs];
    for (int i = 0; i < totalPairs; ++i) {
        kvPairs[i] = Pair<Key, Value>(
            std::make_optional(i + 1),
            std::make_optional((i + 1) * 10)
        );
    }

    const int batchSize = 32;
    for (int i = 0; i < totalPairs; i += batchSize) {
        if (!tree.updateKeys(kvPairs + i, batchSize)) {
            printf("Error: Insertion failed for batch starting at index %d.\n", i);
            return;
        }
        printf("Successfully inserted batch starting at index %d.\n", i);
    }

    const int numKeysToDelete = 32;
    Pair<Key, Value> keysToDelete[numKeysToDelete];
    for (int i = 0; i < numKeysToDelete; ++i) {
        keysToDelete[i] = Pair<Key, Value>(
            std::make_optional((i + 1) * 2),
            std::nullopt  // Tombstone value
        );
    }

    if (!tree.updateKeys(keysToDelete, numKeysToDelete)) {
        printf("Error: Deletion failed for keys.\n");
        return;
    }
    printf("Successfully deleted %d keys.\n", numKeysToDelete);
    tree.printAllLevels();

    printf("\nTesting correctness of deleted key queries:\n");
    int numCorrect = 0;
    bool foundFlags[numKeysToDelete];
    Value results[numKeysToDelete];
    Key queryKeys[numKeysToDelete];

    for (int i = 0; i < numKeysToDelete; ++i) {
        queryKeys[i] = *keysToDelete[i].first;
    }

    tree.queryKeys(queryKeys, numKeysToDelete, results, foundFlags);

    for (int i = 0; i < numKeysToDelete; ++i) {
        if (!foundFlags[i]) {
            numCorrect++;
        } else {
            printf("Error: Deleted key %d was found with value %d.\n", 
                   queryKeys[i], 
                   results[i]);
        }
    }

    printf("\nCorrectly handled %d out of %d deleted keys.\n", numCorrect, numKeysToDelete);
    if (numCorrect == numKeysToDelete) {
        printf("Test passed: All deleted keys were correctly identified.\n");
    } else {
        printf("Test failed: Some deleted keys were not correctly identified.\n");
    }

    // Test non-deleted keys
    printf("\nTesting correctness of non-deleted key queries:\n");
    numCorrect = 0;
    const int numNonDeletedKeys = 32;
    Key nonDeletedKeys[numNonDeletedKeys];
    Value expectedValues[numNonDeletedKeys];
    bool foundFlagsNonDeleted[numNonDeletedKeys];
    Value resultsNonDeleted[numNonDeletedKeys];

    for (int i = 0; i < numNonDeletedKeys; ++i) {
        nonDeletedKeys[i] = (i * 2) + 1;
        expectedValues[i] = nonDeletedKeys[i] * 10;
    }

    tree.queryKeys(nonDeletedKeys, numNonDeletedKeys, resultsNonDeleted, foundFlagsNonDeleted);

    for (int i = 0; i < numNonDeletedKeys; ++i) {
        if (foundFlagsNonDeleted[i] && resultsNonDeleted[i] == expectedValues[i]) {
            numCorrect++;
        } else {
            printf("Error: Key %d expected value %d but got %d.\n", 
                   nonDeletedKeys[i], 
                   expectedValues[i], 
                   resultsNonDeleted[i]);
        }
    }

    printf("\nCorrectly retrieved %d out of %d non-deleted key-value pairs.\n", 
           numCorrect, 
           numNonDeletedKeys);
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
