#include <bits/stdc++.h>
#include <lsm.cuh>
#include <cuda.h>
#include <merge.cuh>
#include <bitonicSort.cuh>

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
    Key keysToDelete[numKeysToDelete];
    for (int i = 0; i < numKeysToDelete; ++i) {
        keysToDelete[i] = ((i + 1) * 2);
    }

    if (!tree.deleteKeys(keysToDelete, numKeysToDelete)) {
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
        queryKeys[i] = keysToDelete[i];
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

void testMergeWithTombstones() {
    using Key = int;
    using Value = int;

    const int size1 = 5;
    const int size2 = 5;

    // Define arrays with duplicates and tombstones (nullopt values)
    Pair<Key, Value> h_arr1[size1] = {
        {1, std::nullopt}, {3, std::nullopt}, {5, 50}, {7, std::nullopt}, {9, 70}
    };
    Pair<Key, Value> h_arr2[size2] = {
        {2, 20}, {3, 30}, {5, std::nullopt}, {8, 80}, {9, std::nullopt}
    };

    // Pair<Key, Value> h_arr1[size1] = {
    //     {1, 10}, {3, 40}, {5, 50}, {7, 60}, {9, 70}
    // };
    // Pair<Key, Value> h_arr2[size2] = {
    //     {2, 20}, {3, 30}, {5, 40}, {8, 80}, {9, 80}
    // };
    Pair<Key, Value> h_merged[size1 + size2];

    // Allocate device memory and copy data from host to device
    Pair<Key, Value> *d_arr1, *d_arr2;
    cudaMalloc(&d_arr1, size1 * sizeof(Pair<Key, Value>));
    cudaMalloc(&d_arr2, size2 * sizeof(Pair<Key, Value>));
    cudaMemcpy(d_arr1, h_arr1, size1 * sizeof(Pair<Key, Value>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr2, h_arr2, size2 * sizeof(Pair<Key, Value>), cudaMemcpyHostToDevice);

    // Call the merge function
    Pair<Key, Value> *d_merged = merge(d_arr1, size1, d_arr2, size2);

    // Copy the result back to the host
    cudaMemcpy(h_merged, d_merged, (size1 + size2) * sizeof(Pair<Key, Value>), cudaMemcpyDeviceToHost);

    // Expected result including duplicates and tombstones
    std::vector<Pair<Key, Value>> expectedResult = {
        {1, std::nullopt}, {2, 20}, {3, std::nullopt}, {3, 30}, {5, 50}, {5, std::nullopt}, {7, std::nullopt}, {8, 80}, {9, 70}, {9, std::nullopt}
    };
    // std::vector<Pair<Key, Value>> expectedResult = {
    //     {1, 10}, {2, 20}, {3, 40}, {3, 30}, {5, 50}, {5, 40}, {7, 60}, {8, 80}, {9, 70}, {9, 80}
    // };

    // Print the merged array and expected result side by side
    std::cout << "\nMerged array vs. Expected result:\n";
    std::cout << std::setw(20) << "Merged Array" << std::setw(25) << "Expected Result\n";
    std::cout << "-----------------------------------------------------------\n";
    for (size_t i = 0; i < expectedResult.size(); ++i) {
        std::string mergedKey = h_merged[i].first.has_value() ? std::to_string(h_merged[i].first.value()) : "nullopt";
        std::string mergedValue = h_merged[i].second.has_value() ? std::to_string(h_merged[i].second.value()) : "nullopt";
        std::string expectedKey = expectedResult[i].first.has_value() ? std::to_string(expectedResult[i].first.value()) : "nullopt";
        std::string expectedValue = expectedResult[i].second.has_value() ? std::to_string(expectedResult[i].second.value()) : "nullopt";

        std::cout << std::setw(10) << "(" + mergedKey + ", " + mergedValue + ")"
                  << std::setw(20) << "(" + expectedKey + ", " + expectedValue + ")\n";
    }

    // Validate the result
    bool isValid = true;
    for (size_t i = 0; i < expectedResult.size(); ++i) {
        if (h_merged[i].first != expectedResult[i].first || h_merged[i].second != expectedResult[i].second) {
            isValid = false;
            std::cout << "Mismatch at index " << i << ": "
                      << "Expected (" << (expectedResult[i].first.has_value() ? std::to_string(expectedResult[i].first.value()) : "nullopt")
                      << ", "
                      << (expectedResult[i].second.has_value() ? std::to_string(expectedResult[i].second.value()) : "nullopt")
                      << ") but got ("
                      << (h_merged[i].first.has_value() ? std::to_string(h_merged[i].first.value()) : "nullopt")
                      << ", "
                      << (h_merged[i].second.has_value() ? std::to_string(h_merged[i].second.value()) : "nullopt")
                      << ")\n";
        }
    }

    if (isValid) {
        std::cout << "\nTest passed: Merge with duplicates and tombstones handled correctly.\n";
    } else {
        std::cout << "\nTest failed: Merge with duplicates and tombstones produced incorrect results.\n";
    }

    // Free device memory
    cudaFree(d_arr1);
    cudaFree(d_arr2);
    cudaFree(d_merged);
}

void testBitonicSortWithNulloptGPU() {
    const long int N = 8;
    using Key = int;
    using Value = int;
    using DataType = Pair<Key, Value>;

    // Initialize array with key-value pairs, including some nullopt values
    DataType h_arr[N] = {
        {1, 10}, {4, std::nullopt}, {4, 50}, {2, 30}, {2, std::nullopt}, {6, 60}, {7, std::nullopt}, {11, 80}
    };

    // Expected sorted result with `nullopt` entries prioritized
    DataType expected[N] = {
        {1, 10}, {2, std::nullopt}, {2, 30}, {4, std::nullopt}, {4, 50}, {6, 60}, {7, std::nullopt}, {11, 80}
    };

    // Allocate device memory
    DataType* d_arr;
    cudaMalloc(&d_arr, N * sizeof(DataType));

    // Copy data to device
    cudaMemcpy(d_arr, h_arr, N * sizeof(DataType), cudaMemcpyHostToDevice);

    // Perform GPU bitonic sort
    bitonicSortGPU<Key, Value>(d_arr, N);

    // Copy sorted data back to host
    cudaMemcpy(h_arr, d_arr, N * sizeof(DataType), cudaMemcpyDeviceToHost);

    // Print the sorted array
    std::cout << "Sorted array (GPU):\n";
    for (long int i = 0; i < N; i++) {
        std::cout << "("
                  << (h_arr[i].first.has_value() ? std::to_string(h_arr[i].first.value()) : "nullopt")
                  << ", "
                  << (h_arr[i].second.has_value() ? std::to_string(h_arr[i].second.value()) : "nullopt")
                  << ") ";
    }
    std::cout << "\n";

    // Verify the sorted array against the expected result
    bool isCorrect = true;
    for (long int i = 0; i < N; i++) {
        if (h_arr[i].first != expected[i].first || h_arr[i].second != expected[i].second) {
            isCorrect = false;
            std::cout << "Mismatch at index " << i << ": "
                      << "Expected ("
                      << (expected[i].first.has_value() ? std::to_string(expected[i].first.value()) : "nullopt")
                      << ", "
                      << (expected[i].second.has_value() ? std::to_string(expected[i].second.value()) : "nullopt")
                      << ") but got ("
                      << (h_arr[i].first.has_value() ? std::to_string(h_arr[i].first.value()) : "nullopt")
                      << ", "
                      << (h_arr[i].second.has_value() ? std::to_string(h_arr[i].second.value()) : "nullopt")
                      << ")\n";
        }
    }

    if (isCorrect) {
        std::cout << "Test passed: GPU bitonic sort handled `nullopt` priority correctly.\n";
    } else {
        std::cout << "Test failed: Mismatches found in the sorted result.\n";
    }

    // Clean up resources
    cudaFree(d_arr);
}

void runTestCountKeys() {
    using Key = int;
    using Value = int;
    const int numLevels = 4;
    const int bufferSize = 32;

    // Initialize LSM tree
    lsmTree<Key, Value> tree(numLevels, bufferSize);

    const int totalPairs = 96;
    Pair<Key, Value> kvPairs[totalPairs];

    // Populate key-value pairs with unique keys in increasing order
    for (int i = 0; i < totalPairs; ++i) {
        kvPairs[i] = Pair<Key, Value>(
            std::make_optional(i + 1),
            std::make_optional((i + 1) * 10)
        );
    }

    // Insert key-value pairs in batches
    const int batchSize = 32;
    for (int i = 0; i < totalPairs; i += batchSize) {
        if (!tree.updateKeys(kvPairs + i, batchSize)) {
            printf("Error: Insertion failed for batch starting at index %d.\n", i);
            return;
        }
    }

    printf("\nTesting countKeys function:\n");

    // Define queries with ranges (k1, k2)
    const int numQueries = 5;
    Key k1[numQueries] = {1, 10, 20, 30, 40};
    Key k2[numQueries] = {10, 20, 30, 40, 50};
    int counts[numQueries];

    // Call the countKeys function
    tree.countKeys(k1, k2, numQueries, counts);

    // Expected counts for each range
    int expectedCounts[numQueries] = {10, 11, 11, 11, 11};

    // Validate the results
    bool isCorrect = true;
    for (int i = 0; i < numQueries; ++i) {
        if (counts[i] != expectedCounts[i]) {
            isCorrect = false;
            printf("Error: For range (%d, %d), expected count %d but got %d.\n",
                   k1[i], k2[i], expectedCounts[i], counts[i]);
        }
    }

    if (isCorrect) {
        printf("Test passed: countKeys function returned correct counts for all queries.\n");
    } else {
        printf("Test failed: countKeys function returned incorrect counts for some queries.\n");
    }
}

int main() {
/*    printf("Running Test with Unique Keys:\n");
    runTestWithUniqueKeys();

    printf("\nRunning Test with Duplicate Keys:\n");
    runTestWithDuplicateKeys();

    printf("\nRunning Test with Deleted Keys:\n");
    runTestWithDeletedKeys();

    printf("\nRunning test for merge with tombstones:\n");
    testMergeWithTombstones();

    printf("\nRunning test for sort with nullopt:\n");
    testBitonicSortWithNulloptGPU();*/

    printf("\nRunning test for count keys with no tombstone");
    runTestCountKeys();
    return 0;
}

