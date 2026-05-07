/*
 * CUDA Vector Addition
 * Compile: nvcc -o vector_add 4_1_vector_add.cu
 * Run    : ./vector_add
 */
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void vectorAdd(int* A, int* B, int* C, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        C[tid] = A[tid] + B[tid];
    }
}

void print(int* vector, int size) {
    for (int i = 0; i < size; i++) {
        cout << vector[i] << " ";
    }
    cout << endl;
}

int main() {
    int N = 1000000;
    size_t bytes = N * sizeof(int);

    // Allocate host memory
    int* h_A = new int[N];
    int* h_B = new int[N];
    int* h_C = new int[N];

    // Initialize vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    cout << "Vector Size: " << N << endl << endl;
    cout << "First 5 elements of A: "; 
    for(int i = 0; i < 5; i++) cout << h_A[i] << " "; 
    cout << endl;
    
    cout << "First 5 elements of B: "; 
    for(int i = 0; i < 5; i++) cout << h_B[i] << " "; 
    cout << endl << endl;

    // Allocate device memory
    int* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    cout << "Threads per block: " << threadsPerBlock << endl;
    cout << "Blocks per grid: " << blocksPerGrid << endl << endl;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cout << "First 5 elements of C (A+B): ";
    for(int i = 0; i < 5; i++) cout << h_C[i] << " ";
    cout << endl << endl;

    // Verify result
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            correct = false;
            break;
        }
    }
    cout << "Result: " << (correct ? "CORRECT" : "INCORRECT") << endl;

    // Cleanup
    delete[] h_A; delete[] h_B; delete[] h_C;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}