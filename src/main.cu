#include <bits/stdc++.h>
#include <lsm.cuh>
#include <cuda.h>
#include <iostream>
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

// Helper function to test `countKeys` with duplicates and tombstones
void testCountKeysWithDuplicatesAndTombstones() {
    using Key = int;
    using Value = int;
    const int numLevels = 4;
    const int bufferSize = 32;
    const int totalKeys = 96;
    const int numQueries = 2;

    // Define the LSM tree
    lsmTree<Key, Value> tree(numLevels, bufferSize);
    Pair<Key, Value> kvPairs[bufferSize];

    // Step 1: Insert the initial batch of key-value pairs
    for (int i = 0; i < bufferSize; ++i) {
        kvPairs[i] = Pair<Key, Value>(std::make_optional(i), std::make_optional(i * 10));
    }

    if (!tree.updateKeys(kvPairs, bufferSize)) {
        std::cerr << "Error: Insertion failed for the initial batch.\n";
        return;
    }
    std::cout << "Successfully inserted the initial batch of keys.\n";

    // Step 2: Insert a batch of duplicate keys with updated values
    for (int i = 0; i < bufferSize; ++i) {
        kvPairs[i] = Pair<Key, Value>(std::make_optional(i), std::make_optional(1000 + i * 100));
    }

    if (!tree.updateKeys(kvPairs, bufferSize)) {
        std::cerr << "Error: Insertion failed for the batch of duplicates.\n";
        return;
    }
    std::cout << "Successfully inserted the batch of duplicates with updated values.\n";

    // Step 3: Insert a batch with tombstones (simulating deletions)
    for (int i = 0; i < bufferSize; ++i) {
        if (i % 5 == 0) { // Mark keys divisible by 5 as deleted
            kvPairs[i] = Pair<Key, Value>(std::make_optional(i), std::nullopt);
        } else {
            kvPairs[i] = Pair<Key, Value>(std::make_optional(i), std::make_optional(2000 + i * 100));
        }
    }

    if (!tree.updateKeys(kvPairs, bufferSize)) {
        std::cerr << "Error: Insertion failed for the batch with tombstones.\n";
        return;
    }
    std::cout << "Successfully inserted the batch with tombstones.\n";

    // Print the LSM tree levels after insertion
    tree.printAllLevels();

    // Define keys for lower and upper bound queries
    Key h_k1[numQueries] = {10, 20};
    Key h_k2[numQueries] = {30, 40};

    // Allocate device memory for input keys
    Key *d_k1, *d_k2;
    cudaMalloc(&d_k1, numQueries * sizeof(Key));
    cudaMalloc(&d_k2, numQueries * sizeof(Key));
    cudaMemcpy(d_k1, h_k1, numQueries * sizeof(Key), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k2, h_k2, numQueries * sizeof(Key), cudaMemcpyHostToDevice);

    // Allocate host memory for output counts
    int h_counts[numQueries];

    // Call `countKeys` method
    tree.countKeys(d_k1, d_k2, numQueries, h_counts);

    // Expected results considering tombstones (keys divisible by 5 are deleted)
    int expectedCounts[numQueries] = {16, 9}; // Adjusted counts excluding keys with tombstones

    // Validate the results
    std::cout << "\nTesting `countKeys` method with duplicates and tombstones:\n";
    bool isCorrect = true;
    for (int queryId = 0; queryId < numQueries; ++queryId) {
        std::cout << "Query " << queryId << " -> Count: " << h_counts[queryId] << "\n";
        if (h_counts[queryId] != expectedCounts[queryId]) {
            std::cerr << "Error: Mismatch for Query " << queryId
                      << ". Expected " << expectedCounts[queryId]
                      << " but got " << h_counts[queryId] << ".\n";
            isCorrect = false;
        }
    }

    if (isCorrect) {
        std::cout << "Test passed: `countKeys` method correctly handled duplicates and tombstones.\n";
    } else {
        std::cout << "Test failed: Mismatches found in the `countKeys` results.\n";
    }

    // Free device memory
    cudaFree(d_k1);
    cudaFree(d_k2);
}


void testLSMTreePerformance(const std::vector<int>& testSizes) {
    using Key = int;
    using Value = int;

    for (int numUpdates : testSizes) {
        const int numLevels = 4;
        const int bufferSize = numUpdates/4;

        if (numUpdates <= 0 || numUpdates % bufferSize != 0 || (numUpdates & (numUpdates - 1)) != 0) {
            printf("Error: Invalid input %d. Skipping.\n", numUpdates);
            continue;
        }

        // Measure initialization time
        auto initStart = std::chrono::high_resolution_clock::now();
        lsmTree<Key, Value> tree(numLevels, bufferSize);
        auto initEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> initElapsed = initEnd - initStart;
        printf("Initialized LSM tree with %d levels and buffer size %d in %.6f seconds.\n",
               numLevels, bufferSize, initElapsed.count());

        // Generate random key-value pairs
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<Key> keyDist(1, 1000);
        std::uniform_int_distribution<Value> valueDist(1, 10000);

        Pair<Key, Value>* kvPairs = new Pair<Key, Value>[numUpdates];
        for (int i = 0; i < numUpdates; ++i) {
            kvPairs[i] = Pair<Key, Value>(
                std::make_optional(keyDist(gen)),
                std::make_optional(valueDist(gen))
            );
        }

        // Measure update time
        const int numBatches = numUpdates / bufferSize;
        auto updateStart = std::chrono::high_resolution_clock::now();
        for (int batch = 0; batch < numBatches; ++batch) {
            // Pass a pointer to the batch start and its size
            if (!tree.updateKeys(&kvPairs[batch * bufferSize], bufferSize)) {
                printf("Error: Update failed for batch %d.\n", batch);
                delete[] kvPairs;
                return;
            }
        }
        auto updateEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> updateElapsed = updateEnd - updateStart;
        printf("Updated %d keys in batches of %d in %.6f seconds.\n",
               numUpdates, bufferSize, updateElapsed.count());

        // Generate random keys for querying
        Key* keysToQuery = new Key[numUpdates / 2];
        for (int i = 0; i < numUpdates / 2; ++i) {
            keysToQuery[i] = keyDist(gen);
        }

        // Prepare results and flags
        Value* queryResults = new Value[numUpdates / 2];
        bool* queryFlags = new bool[numUpdates / 2];

        // Measure query time
        auto queryStart = std::chrono::high_resolution_clock::now();
        tree.queryKeys(keysToQuery, numUpdates / 2, queryResults, queryFlags);
        auto queryEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> queryElapsed = queryEnd - queryStart;
        printf("Queried %d keys in %.6f seconds.\n", numUpdates / 2, queryElapsed.count());

        // Free allocated memory
        delete[] kvPairs;
        delete[] keysToQuery;
        delete[] queryResults;
        delete[] queryFlags;
    }
}

int main() {
    // printf("Running Test with Unique Keys:\n");
    // runTestWithUniqueKeys();

    // printf("\nRunning Test with Duplicate Keys:\n");
    // runTestWithDuplicateKeys();

    // printf("\nRunning Test with Deleted Keys:\n");
    // runTestWithDeletedKeys();

    // printf("\nRunning test for merge with tombstones:\n");
    // testMergeWithTombstones();

    // printf("\nRunning test for sort with nullopt:\n");
    // testBitonicSortWithNulloptGPU();

    printf("Running test for countKeys method with duplicates and tombstones:\n");
    testCountKeysWithDuplicatesAndTombstones();

    // std::vector<int> testSizes = {16, 256, 4096, 65536, 1048576, 16777216};
    // testLSMTreePerformance(testSizes);
    return 0;
}


