#include <iostream>
#include <cstring>
#include "lsm.cuh"

int main() {
    int numLevels = 3;      // Number of levels in the tree
    int bufferSize = 4;     // Buffer size (number of key-value pairs per level)
    bool sortStable = true; // Use stable sort

    // Initialize the LSM tree
    lsmTree<float, char*> tree(numLevels, bufferSize, sortStable);

    Pair<float, char*> keyValuePairs[] = {{1.1f, "value1"},{2.2f, "value2"},{3.3f, "value3"},{4.4f, "value4"}};

    tree.updateKeys(keyValuePairs, bufferSize);
    tree.printAllLevels();

    float queryKeys[] = {3.3f, 1.1f, 7.7f};
    int numQueries = sizeof(queryKeys) / sizeof(queryKeys[0]);

    char* results[numQueries];
    bool foundFlags[numQueries];

    tree.queryKeys(queryKeys, numQueries, results, foundFlags);

    // Print query results
    for (int i = 0; i < numQueries; i++) {
        if (foundFlags[i]) {
            std::cout << "Key: " << queryKeys[i] << " -> Value: " << results[i] << std::endl;
        } else {
            std::cout << "Key: " << queryKeys[i] << " -> Not Found" << std::endl;
        }
    }

    return 0;
}

