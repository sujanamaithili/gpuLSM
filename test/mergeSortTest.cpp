#include "merge_sort.cuh"
#include <iostream>
#include <cstdlib>

int main() 
{
    const int N = 1 << 5;
    int* arr = new int[N];
    for (int i = 0; i < N; i++) 
    {
        arr[i] = rand() % N;
    }

    mergeSort(arr, N);

    bool sorted = true;
    for (int i = 1; i < N; i++) 
    {
        if (arr[i - 1] > arr[i]) 
        {
            sorted = false;
            break;
        }
    }

    if (sorted) 
        std::cout << "Array is sorted correctly." << std::endl;
    else 
        std::cout << "Array is not sorted correctly." << std::endl;

    delete[] arr;
    return 0;
}
