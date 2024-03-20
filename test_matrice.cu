#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

struct GlobalVarMatrix{
    float *matriceLeftd;
    float *matriceRightd;
    float *matriceResultd;
};

__global__ void matrixMulKernel(float *matriceResultd, float *matriceLeftd, float *matriceRightd, int width) {
    // Identifiant de thread à deux dimensions, comme la matrice
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // Pvaleur sert au stockage de la valeur calculée par le thread
    float pResult = 0;
    for (int i = 0; i < width; ++i) {
        float mdElement = matriceLeftd[ty * width + i];
        float ndElement = matriceRightd[i * width + tx];
        pResult += mdElement * ndElement;
    }
    // Écrit la valeur calculée dans la matrice de résultat
    // Chaque thread ne peut écrire qu'une valeur !
    matriceResultd[ty * width + tx] = pResult;
}

void matrixMulOnDevice(float *matriceResult, float *matriceLeft, float *matriceRight, int width) {
    // Calcul de la taille des matrices
    int size = width * width * sizeof(float);
    // Allocation des matrices et leur remplissage
    GlobalVarMatrix globalVarMatrix;
    cudaMalloc(&globalVarMatrix.matriceLeftd, size);
    cudaMemcpy(globalVarMatrix.matriceLeftd, matriceLeft, size, cudaMemcpyHostToDevice);
    cudaMalloc(&globalVarMatrix.matriceRightd, size);
    cudaMemcpy(globalVarMatrix.matriceRightd, matriceRight, size, cudaMemcpyHostToDevice);
    // Allocation de la matrice de résultat
    cudaMalloc(&globalVarMatrix.matriceResultd, size);
    // Multiplication d'une seule matrice
    dim3 dimGrid(1, 1);
    // Matrice carrée
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    // Produit matriciel proprement dit
    matrixMulKernel<<<dimGrid, dimBlock>>>(globalVarMatrix.matriceResultd, globalVarMatrix.matriceLeftd, globalVarMatrix.matriceRightd, width);
    // Récupération du résultat du calcul
    cudaMemcpy(matriceResult, globalVarMatrix.matriceResultd, size, cudaMemcpyDeviceToHost);
    // Destruction des matrices, désormais inutilisées
    cudaFree(globalVarMatrix.matriceLeftd);
    cudaFree(globalVarMatrix.matriceRightd);
    cudaFree(globalVarMatrix.matriceResultd);
}

/// Fonction qui initialise le premier terme de la série pseudo-aléatoire
void initRandom() {
    time_t t = time(NULL);
    srand(t);
}

/// Fonction qui tire un nombre aléatoire entre deux bornes
float getRandFloat(float inf, float sup) {
    return inf + (((float) rand()) * (sup - inf)) / ((float) RAND_MAX);
}

/// Fonction qui initialise une matrice carrée avec des nombres aléatoires
void initRandomMatrix(float *matrix, size_t size, float inf, float sup) {
    if (matrix == NULL) return;
    for (size_t i = 0; i < size * size; ++i) {
        matrix[i] = getRandFloat(inf, sup);
    }
}

/// Affiche une matrice carrée
void printMatrix(float *matrix, int width) {
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            printf("%.2f ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    initRandom();
    clock_t temps = clock();
    float chrono;
    int width = 10;  // Défini la taille de la matrice
    int size = width * width * sizeof(float);
    // On alloue les matrices
    float *matriceLeft = (float *) malloc(size);
    float *matriceRight = (float *) malloc(size);
    float *matriceResult = (float *) malloc(size);
    // On initialise les matrices aléatoirement
    initRandomMatrix(matriceLeft, width, -10.0, 10.0);
    initRandomMatrix(matriceRight, width, -10.0, 10.0);
    temps = clock() - temps;
    chrono = ((float) temps) / ((float) CLOCKS_PER_SEC);
    printf("Matrice A :\n");
    printMatrix(matriceLeft, width);
    printf("\nMatrice B :\n");
    printMatrix(matriceRight, width);
    printf("\nTemps de l'initialisation : %fs\n", chrono);
    temps = clock();
    // On appelle la fonction qui fait la multiplication à notre place
    matrixMulOnDevice(matriceResult, matriceLeft, matriceRight, width);
    temps = clock() - temps;
    chrono = ((float) temps) / ((float) CLOCKS_PER_SEC);
    printf("\nRésultat de la multiplication :\n");
    printMatrix(matriceResult, width);
    printf("\nTemps du calcul de la multiplication : %fs\n", chrono);
    // On désalloue les matrices
    free(matriceResult);
    free(matriceRight);
    free(matriceLeft);
    return 0;
}
