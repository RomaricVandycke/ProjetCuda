#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 5 // Taille des matrices

// Kernel pour la multiplication de matrices
__global__ void matrixMultiplication(int *a, int *b, int *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        int sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += a[row * n + i] * b[i + col * n];
        }
        c[row * n + col] = sum;
    }
}

int main() {
    int *a, *b, *c; // Host matrices
    int *d_a, *d_b, *d_c; // Device matrices

    // Allocation mémoire pour les matrices sur le device
    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc((void**)&d_a, N * N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Erreur lors de l'allocation de mémoire pour d_a: %s\n", cudaGetErrorString(cudaStatus));
        return -1;
    }
    cudaStatus = cudaMalloc((void**)&d_b, N * N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Erreur lors de l'allocation de mémoire pour d_b: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_a);
        return -1;
    }
    cudaStatus = cudaMalloc((void**)&d_c, N * N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Erreur lors de l'allocation de mémoire pour d_c: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_a);
        cudaFree(d_b);
        return -1;
    }

    // Allocation mémoire pour les matrices sur l'hôte
    a = (int*)malloc(N * N * sizeof(int));
    b = (int*)malloc(N * N * sizeof(int));
    c = (int*)malloc(N * N * sizeof(int));

    // Initialisation des matrices a et b avec des valeurs aléatoires
    srand(time(NULL));
    for (int i = 0; i < N * N; ++i) {
        a[i] = rand() % 10; // Valeurs entre 0 et 9
        b[i] = rand() % 10; // Valeurs entre 0 et 9
    }

    // Copie des données des matrices de l'hôte au device
    cudaStatus = cudaMemcpy(d_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Erreur lors de la copie des données de a au device: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return -1;
    }
    cudaStatus = cudaMemcpy(d_b, b, N * N * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Erreur lors de la copie des données de b au device: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return -1;
    }

    // Configuration des dimensions de la grille et du bloc
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Appel du kernel pour la multiplication de matrices
    matrixMultiplication<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Copie du résultat de la multiplication du device à l'hôte
    cudaStatus = cudaMemcpy(c, d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Erreur lors de la copie du résultat de c au host: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return -1;
    }

    // Affichage du résultat
    printf("Matrix A:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d ", a[i * N + j]);
        }
        printf("\n");
    }

    printf("\nMatrix B:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d ", b[i * N + j]);
        }
        printf("\n");
    }

    printf("\nResult Matrix C:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d ", c[i * N + j]);
        }
        printf("\n");
    }

    // Libération de la mémoire
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
