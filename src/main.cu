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

    // Define 32 key-value pairs for insertion
    Pair<Key, Value> kvPairs[96];
    for (int i = 0; i < 96; ++i) {
        kvPairs[i] = {96 - (i + 1), (96 - (i + 1)) * 10};
    }
    
    //std::random_device rd;  // Seed for random number generation
    //std::mt19937 gen(rd()); // Mersenne Twister random number generator
    //std::shuffle(std::begin(kvPairs), std::end(kvPairs), gen);

    // Insert the batch of 32 key-value pairs
    if (!tree.updateKeys(kvPairs, 96)) {
        printf("Error: Insertion failed.\n");
        return 1;
    }

    // Print the LSM tree structure after the insertion
    printf("\nLSM Tree Structure after inserting 32 key-value pairs:\n");
    tree.printAllLevels();

    return 0;
}
