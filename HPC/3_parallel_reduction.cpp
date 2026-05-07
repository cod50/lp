/*
 * Parallel Reduction: Min, Max, Sum, Average using OpenMP
 * Compile: g++ -fopenmp -o parallel_reduction 3_parallel_reduction.cpp
 * Run    : ./parallel_reduction
 */

#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <ctime>

using namespace std;

int main() {
    int n = 1000000;
    int* arr = new int[n];
    
    // Seed random number generator and populate array
    srand(time(0));
    for(int i = 0; i < n; i++) {
        arr[i] = rand() % 1000 + 1;  // Random numbers 1-1000
    }
    
    cout << "Array Size: " << n << endl;
    cout << "Number of Threads: " << omp_get_max_threads() << endl << endl;
    
    // --- Minimum Reduction ---
    int minval = arr[0];
    #pragma omp parallel for reduction(min : minval)
    for(int i = 0; i < n; i++) {
        if(arr[i] < minval) minval = arr[i];
    }
    
    // --- Maximum Reduction ---
    int maxval = arr[0];
    #pragma omp parallel for reduction(max : maxval)
    for(int i = 0; i < n; i++) {
        if(arr[i] > maxval) maxval = arr[i];
    }
    
    // --- Sum Reduction ---
    long long sum = 0;
    #pragma omp parallel for reduction(+ : sum)
    for(int i = 0; i < n; i++) {
        sum += arr[i];
    }
    
    // --- Average ---
    double average = (double)sum / n;
    
    // Display Results
    cout << "The minimum value is: " << minval << endl;
    cout << "The maximum value is: " << maxval << endl;
    cout << "The summation is: " << sum << endl;
    cout << "The average is: " << average << endl;
    
    delete[] arr;
    return 0;
}