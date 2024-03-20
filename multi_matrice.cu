#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

typedef float basetype;  // Vous pouvez remplacer "float" par le type de données souhaité

typedef struct resnfo {
    int seconds;        // Secondes
    int microseconds;   // Microsecondes
    // D'autres membres peuvent être ajoutés au besoin
} resnfo;


// Fonctions de multiplication de matrices en C
void MultMat_CPU(const basetype arrayA[], const basetype arrayB[], 
      basetype arrayR[], const unsigned int n)
{
    unsigned int i, j, k;
    basetype res;

    for(i = 0; i < n; i++) 
        for(j= 0; j<n; j++){
            res = 0;
            for(k=0; k<n; k++)
                res += arrayA[i*n+k] * arrayB[k*n+j];
 
            arrayR[i*n+j]= res;
        }
}

__global__ void multmat_kernel_cuda(const basetype *const mA, 
      const basetype *const mB, 
      basetype *const mR, const int n)
{
    //2D Thread ID
    int row = (blockIdx.x * blockDim.x+ threadIdx.x)/n;
    int column = (blockIdx.x * blockDim.x+ threadIdx.x) % n;

    //Pvalue stores the Pd element that is computed by the thread
    float Pvalue = 0;

    for(int k = 0; k < n; ++k) {
        float Mdelement = mA[row*n + k];
        float Ndelement = mB[k*n + column];
        Pvalue += (Mdelement*Ndelement);
    }

    mR[row*n + column] = Pvalue;
}

void multmat_GPU(const basetype arrayA[], const basetype arrayB[], 
      basetype arrayR[], const unsigned int n, 
      const unsigned int blk_size, 
      resnfo *const start, resnfo *const end)
{
    // Número de bytes de cada uno de nuestros vectores
    unsigned int numBytes = n * n* sizeof(basetype);

    // Reservamos memoria global del device (GPU) para nuestros 
    // arrays y los copiamos
    basetype *cA;
    cudaMalloc((void **) &cA, numBytes);
    cudaMemcpy(cA, arrayA, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU

    basetype *cB;
    cudaMalloc((void **) &cB, numBytes);
    cudaMemcpy(cB, arrayB, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU

    basetype *cR;
    cudaMalloc((void **) &cR, numBytes);
    cudaMemset(cR, 0, numBytes); // Inicializamos (a 0) array para el resultado

    // Bloque unidimensional de hilos (*blk_size* hilos)
    dim3 dimBlock(blk_size);

    // Rejilla unidimensional (*ceil(n/blk_size)* bloques)
    dim3 dimGrid((n*n + dimBlock.x - 1) / dimBlock.x);

    // Lanzamos ejecución del kernel en la GPU *r* veces
    timestamp(start);            // Medimos tiempo de cálculo en GPU
  
    multmat_kernel_cuda<<<dimGrid, dimBlock>>>(cA, cB, cR, n);
    
    cudaDeviceSynchronize();
    timestamp(end);

    cudaMemcpy(arrayR, cR, numBytes, cudaMemcpyDeviceToHost); // GPU -> CPU
}

int main() {
    // Déclaration des matrices et des autres variables nécessaires
    const int n = 4;  // Taille des matrices
    basetype matrixA[n*n], matrixB[n*n], result_CPU[n*n], result_GPU[n*n];

    // Initialisation des matrices A et B avec des valeurs aléatoires
    for(int i = 0; i < n*n; i++) {
        matrixA[i] = rand() % 10;  // Valeurs aléatoires entre 0 et 9
        matrixB[i] = rand() % 10;
    }

    // Appel de la fonction pour la multiplication de matrices sur le CPU
    MultMat_CPU(matrixA, matrixB, result_CPU, n);

    // Appel de la fonction pour la multiplication de matrices sur le GPU
    resnfo start, end;
    multmat_GPU(matrixA, matrixB, result_GPU, n, 256, &start, &end);

    // Affichage des résultats
    printf("Résultat de la multiplication de matrices sur le CPU :\n");
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            printf("%f ", result_CPU[i*n + j]);
        }
        printf("\n");
    }
    printf("\n");

    printf("Résultat de la multiplication de matrices sur le GPU :\n");
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            printf("%f ", result_GPU[i*n + j]);
        }
        printf("\n");
    }
    printf("\n");

    return 0;
}

