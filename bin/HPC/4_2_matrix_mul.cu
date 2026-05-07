/*
 * CUDA Matrix Multiplication
 * Compile: nvcc -o matrix_mul 4_2_matrix_mul.cu
 * Run    : ./matrix_mul
 */
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void matrixMultiply(int* A, int* B, int* C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        int sum = 0;
        for (int i = 0; i < size; i++) {
            sum += A[row * size + i] * B[i * size + col];
        }
        C[row * size + col] = sum;
    }
}

void initialize(int* matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = rand() % 10;
    }
}

void print(int* matrix, int size) {
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            cout << matrix[row * size + col] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

int main() {
    int N = 512;  // Matrix size NxN
    size_t bytes = N * N * sizeof(int);

    // Allocate host memory
    int* h_A = new int[N * N];
    int* h_B = new int[N * N];
    int* h_C = new int[N * N];

    // Initialize matrices
    initialize(h_A, N);
    initialize(h_B, N);

    cout << "Matrix Size: " << N << "x" << N << endl << endl;

    // Print sample values (only for small matrices)
    if (N <= 4) {
        cout << "Matrix A:\n"; print(h_A, N);
        cout << "Matrix B:\n"; print(h_B, N);
    }

    // Allocate device memory
    int* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int THREADS = 16;
    int BLOCKS = (N + THREADS - 1) / THREADS;

    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    cout << "Grid: " << BLOCKS << "x" << BLOCKS << " blocks" << endl;
    cout << "Threads: " << THREADS << "x" << THREADS << " per block" << endl << endl;

    matrixMultiply<<<blocks, threads>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    if (N <= 4) {
        cout << "Result Matrix C (A x B):\n";
        print(h_C, N);
    } else {
        cout << "First 5 elements of result: ";
        for(int i = 0; i < 5; i++) cout << h_C[i] << " ";
        cout << endl << endl;
    }

    cout << "Computation complete!" << endl;

    // Cleanup
    delete[] h_A; delete[] h_B; delete[] h_C;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}