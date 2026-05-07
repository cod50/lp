/*
 * Parallel Bubble & Merge Sort using OpenMP
 * Compile: g++ -fopenmp -o parallel_sort 2_parallel_sort.cpp 
 * Run    : ./parallel_sort
 */
#include <iostream>
#include <omp.h>
using namespace std;

// Sequential Bubble Sort
void bubble(int arr[], int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - i - 1; j++)
            if (arr[j] > arr[j + 1]) 
                swap(arr[j], arr[j + 1]);
}

// Parallel Bubble Sort (Odd-Even Sort)
void parallelBubble(int arr[], int n) {
    for (int i = 0; i < n; i++) {
        // Odd phase
        #pragma omp parallel for
        for (int j = 1; j < n; j += 2)
            if (arr[j] < arr[j - 1])
                swap(arr[j], arr[j - 1]);

        // Even phase
        #pragma omp parallel for
        for (int j = 2; j < n; j += 2)
            if (arr[j] < arr[j - 1])
                swap(arr[j], arr[j - 1]);
    }
}

// Merge function for merge sort
void merge(int arr[], int low, int mid, int high) {
    int n1 = mid - low + 1;
    int n2 = high - mid;
    int left[n1], right[n2];

    for (int i = 0; i < n1; i++) left[i] = arr[low + i];
    for (int j = 0; j < n2; j++) right[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = low;
    while (i < n1 && j < n2) {
        if (left[i] <= right[j]) arr[k++] = left[i++];
        else arr[k++] = right[j++];
    }
    while (i < n1) arr[k++] = left[i++];
    while (j < n2) arr[k++] = right[j++];
}

// Sequential Merge Sort
void mergeSort(int arr[], int low, int high) {
    if (low < high) {
        int mid = (low + high) / 2;
        mergeSort(arr, low, mid);
        mergeSort(arr, mid + 1, high);
        merge(arr, low, mid, high);
    }
}

// Parallel Merge Sort
void parallelMergeSort(int arr[], int low, int high) {
    if (low < high) {
        int mid = (low + high) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            parallelMergeSort(arr, low, mid);

            #pragma omp section
            parallelMergeSort(arr, mid + 1, high);
        }
        merge(arr, low, mid, high);
    }
}

void printArray(int arr[], int n) {
    for (int i = 0; i < n; i++) cout << arr[i] << " ";
    cout << endl;
}

int main() {
    int n = 10;
    int arr[n];
    double start_time, end_time;

    cout << "Array Size: " << n << " | Threads: " << omp_get_max_threads() << "\n\n";

    // ========== BUBBLE SORT ==========
    cout << "========== BUBBLE SORT ==========\n";
    
    // Sequential Bubble Sort
    for (int i = 0, j = n; i < n; i++, j--) arr[i] = j;
    start_time = omp_get_wtime();
    bubble(arr, n);
    end_time = omp_get_wtime();
    cout << "Sequential Bubble Sort: " << end_time - start_time << " seconds\n";
    printArray(arr, n);

    // Parallel Bubble Sort
    for (int i = 0, j = n; i < n; i++, j--) arr[i] = j;
    start_time = omp_get_wtime();
    parallelBubble(arr, n);
    end_time = omp_get_wtime();
    cout << "Parallel Bubble Sort: " << end_time - start_time << " seconds\n";
    printArray(arr, n);

    // ========== MERGE SORT ==========
    cout << "\n========== MERGE SORT ==========\n";
    
    // Sequential Merge Sort
    for (int i = 0, j = n; i < n; i++, j--) arr[i] = j;
    start_time = omp_get_wtime();
    mergeSort(arr, 0, n - 1);
    end_time = omp_get_wtime();
    cout << "Sequential Merge Sort: " << end_time - start_time << " seconds\n";
    printArray(arr, n);

    // Parallel Merge Sort
    for (int i = 0, j = n; i < n; i++, j--) arr[i] = j;
    start_time = omp_get_wtime();
    parallelMergeSort(arr, 0, n - 1);
    end_time = omp_get_wtime();
    cout << "Parallel Merge Sort: " << end_time - start_time << " seconds\n";
    printArray(arr, n);

    return 0;
}