#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <unistd.h> // Pour utiliser la fonction sleep

typedef float basetype;

typedef struct resnfo {
    int seconds;
    int microseconds;
} resnfo;

void timestamp(struct timespec *ts) {
    clock_gettime(CLOCK_MONOTONIC, ts);
}

void MultMat_CPU(const basetype arrayA[], const basetype arrayB[], 
      basetype arrayR[], const unsigned int n)
{
    unsigned int i, j, k;
    basetype res;

    for(i = 0; i < n; i++) 
        for(j = 0; j < n; j++){
            res = 0;
            for(k = 0; k < n; k++)
                res += arrayA[i * n + k] * arrayB[k * n + j];
 
            arrayR[i * n + j] = res;
        }
}

__global__ void multmat_kernel_cuda(const basetype *const mA, 
      const basetype *const mB, 
      basetype *const mR, const int n)
{
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / n;
    int column = (blockIdx.x * blockDim.x + threadIdx.x) % n;

    float Pvalue = 0;

    for(int k = 0; k < n; ++k) {
        float Mdelement = mA[row * n + k];
        float Ndelement = mB[k * n + column];
        Pvalue += (Mdelement * Ndelement);
    }

    mR[row * n + column] = Pvalue;
}

void multmat_GPU(const basetype arrayA[], const basetype arrayB[], 
      basetype arrayR[], const unsigned int n, 
      const unsigned int blk_size, 
      resnfo *const start, resnfo *const end)
{
    unsigned int numBytes = n * n * sizeof(basetype);

    basetype *cA;
    cudaMalloc((void **) &cA, numBytes);
    cudaMemcpy(cA, arrayA, numBytes, cudaMemcpyHostToDevice);

    basetype *cB;
    cudaMalloc((void **) &cB, numBytes);
    cudaMemcpy(cB, arrayB, numBytes, cudaMemcpyHostToDevice);

    basetype *cR;
    cudaMalloc((void **) &cR, numBytes);
    cudaMemset(cR, 0, numBytes);

    dim3 dimBlock(blk_size);
    dim3 dimGrid((n * n + dimBlock.x - 1) / dimBlock.x);

    timestamp(start);

    multmat_kernel_cuda<<<dimGrid, dimBlock>>>(cA, cB, cR, n);
    
    cudaDeviceSynchronize();
    timestamp(end);

    cudaMemcpy(arrayR, cR, numBytes, cudaMemcpyDeviceToHost);
}

int main() {
    const int n = 4;
    basetype matrixA[n * n], matrixB[n * n], result_CPU[n * n], result_GPU[n * n];

    for(int i = 0; i < n * n; i++) {
        matrixA[i] = rand() % 10;
        matrixB[i] = rand() % 10;
    }

    MultMat_CPU(matrixA, matrixB, result_CPU, n);

    resnfo start, end;
    multmat_GPU(matrixA, matrixB, result_GPU, n, 256, &start, &end);

    printf("Résultat de la multiplication de matrices sur le CPU :\n");
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


    return 0;
}

