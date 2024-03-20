#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

typedef float basetype;  // Utilisation du type float pour les données

// Structure pour stocker les informations temporelles
typedef struct timespec resnfo;

// Fonction pour obtenir le temps actuel
void timestamp(resnfo *ts) {
    clock_gettime(CLOCK_MONOTONIC, ts);
}

// Fonction pour la multiplication de matrices sur le CPU
void MultMat_CPU(const basetype arrayA[], const basetype arrayB[], 
      basetype arrayR[], const unsigned int n)
{
    unsigned int i, j, k;
    basetype res;

    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            res = 0;
            for(k = 0; k < n; k++)
                res += arrayA[i * n + k] * arrayB[k * n + j];
            arrayR[i * n + j] = res;
        }
    }
}

// Noyau CUDA pour la multiplication de matrices sur le GPU
__global__ void multmat_kernel_cuda(const basetype *const mA, 
      const basetype *const mB, 
      basetype *const mR, const int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        basetype Pvalue = 0;
        for(int k = 0; k < n; ++k) {
            Pvalue += mA[row * n + k] * mB[k * n + col];
        }
        mR[row * n + col] = Pvalue;
    }
}

// Fonction pour la multiplication de matrices sur le GPU
__global__ void multmat_GPU(const basetype arrayA[], const basetype arrayB[], 
      basetype arrayR[], const unsigned int n, 
      const unsigned int blk_size, 
      resnfo *const start, resnfo *const end)
{
    unsigned int numBytes = n * n * sizeof(basetype);

    basetype *cA, *cB, *cR;
    cudaMalloc((void **)&cA, numBytes);
    cudaMalloc((void **)&cB, numBytes);
    cudaMalloc((void **)&cR, numBytes);

    cudaMemcpy(cA, arrayA, numBytes, cudaMemcpyHostToDevice); // cudaMemcpy permet de transférer des données entre le CPU et le GPU
    cudaMemcpy(cB, arrayB, numBytes, cudaMemcpyHostToDevice);

    dim3 dimBlock(blk_size, blk_size);
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (n + dimBlock.y - 1) / dimBlock.y);

    timestamp(start);
    multmat_kernel_cuda<<<dimGrid, dimBlock>>>(cA, cB, cR, n);
    cudaDeviceSynchronize();
    timestamp(end);

    cudaMemcpy(arrayR, cR, numBytes, cudaMemcpyDeviceToHost);

    cudaFree(cA);
    cudaFree(cB);
    cudaFree(cR);
}

int main() {
    const int n = 4;
    basetype matrixA[n * n], matrixB[n * n], result_CPU[n * n], result_GPU[n * n];

    for(int i = 0; i < n * n; i++) {
        matrixA[i] = rand() % 10;
        matrixB[i] = rand() % 10;
    }

    resnfo start, end;

    // Multiplication de matrices sur le CPU
    timestamp(&start);
    MultMat_CPU(matrixA, matrixB, result_CPU, n);
    timestamp(&end);
    printf("Temps d'exécution sur le CPU : %ld ns\n", (end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec));

    // Multiplication de matrices sur le GPU
    multmat_GPU(matrixA, matrixB, result_GPU, n, 16, &start, &end);
    printf("Temps d'exécution sur le GPU : %ld ns\n", (end.tv_sec - start.tv_sec) * 1000000000 + (end.tv_nsec - start.tv_nsec));

    // Affichage des résultats
    printf("\nRésultat de la multiplication de matrices sur le CPU :\n");
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            printf("%f ", result_CPU[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");

    printf("Résultat de la multiplication de matrices sur le GPU :\n");
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            printf("%f ", result_GPU[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");

    return 0;
}
