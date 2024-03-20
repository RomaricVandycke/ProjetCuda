#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(call) { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("CUDA error: %s, line %d\n", cudaGetErrorString(error), __LINE__); \
        exit(1); \
    } \
}

// Fonction pour initialiser la matrice avec des valeurs aléatoires
void initRandomMatrix(float* matrix, int size, float minVal, float maxVal) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = minVal + static_cast<float>(rand()) / static_cast<float>(RAND_MAX / (maxVal - minVal));
    }
}

// Kernel CUDA pour la multiplication de matrices
__global__ void matrixMulKernel(const float* A, const float* B, float* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float value = 0.0f;
        for (int k = 0; k < width; ++k) {
            value += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = value;
    }
}

// Fonction pour la multiplication de matrices sur GPU
void matrixMulOnDevice(float *C, const float *A, const float *B, int width) {
    int size = width * width * sizeof(float);

    // Allocation de mémoire sur le GPU
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C, size));

    // Copie des données du CPU vers le GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

    // Définition de la configuration des blocs et des grilles pour le kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (width + blockSize.y - 1) / blockSize.y);

    // Appel du kernel CUDA pour la multiplication de matrices
    matrixMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, width);

    // Copie des résultats du GPU vers le CPU
    CHECK_CUDA_ERROR(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));

    // Libération de la mémoire allouée sur le GPU
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
}

int main() {
    srand(time(NULL));
    const int width = 4;
    const int size = width * width;

    float *A = (float*)malloc(size * sizeof(float));
    float *B = (float*)malloc(size * sizeof(float));
    float *C = (float*)malloc(size * sizeof(float));

    // Initialisation des matrices avec des valeurs aléatoires
    initRandomMatrix(A, size, -10.0f, 10.0f);
    initRandomMatrix(B, size, -10.0f, 10.0f);

    // Multiplication de matrices sur GPU
    matrixMulOnDevice(C, A, B, width);

    // Affichage des matrices
    printf("Matrice A :\n");
    for (int i = 0; i < size; ++i) {
        printf("%.2f ", A[i]);
        if ((i + 1) % width == 0) printf("\n");
    }
    printf("\n");

    printf("Matrice B :\n");
    for (int i = 0; i < size; ++i) {
        printf("%.2f ", B[i]);
        if ((i + 1) % width == 0) printf("\n");
    }
    printf("\n");

    printf("Résultat de la multiplication :\n");
    for (int i = 0; i < size; ++i) {
        printf("%.2f ", C[i]);
        if ((i + 1) % width == 0) printf("\n");
    }

    // Libération de la mémoire
    free(A);
    free(B);
    free(C);

    return 0;
}
