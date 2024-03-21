#include <stdio.h>
#include <time.h>
#include <math.h>
#include <malloc.h>
#include <stdlib.h>

#include "mlp.h"

// Kernel pour la multiplication de matrices
__global__ void matrixMultiplicationKernel(double *input_matrix, double *d_weight, double *output_matrix, int m, int n, int k) {
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

typedef struct _neuron NEURON;
struct _neuron {
  int layer;

  double * weight;      // table of weights for incoming synapses
  int nbsynapsesin;     // number of incoming synapses

  NEURON ** synapsesin; // table of pointer to the neurons from
                        // which are coming the synapses
  double bias;

  double value;
  double value_prev;
  double error;
  double error_prev;
};

typedef struct _rnn RNN;
struct _rnn {
  int * layersize;

  int nbneurons;
  NEURON * n;
};

typedef struct _config CONFIG;
struct _config {
  int nbneurons;
  int * layersize;
  int nbsynapses;
  int * synapses;
};


CONFIG * createconfig(int * layersize) {
  CONFIG * conf = (CONFIG*)malloc(sizeof(CONFIG));
  int i;
  conf->nbneurons = 0;
  for(i=1; i<layersize[0]+1; i++) conf->nbneurons += layersize[i];
  conf->layersize = (int*)malloc((layersize[0]+1)*sizeof(int));
  for(i=0; i<layersize[0]+1; i++) conf->layersize[i] = layersize[i];

  // Compute the number of synapses:
  conf->nbsynapses = 0;
  for(i=1; i<layersize[0]; i++) conf->nbsynapses += layersize[i] * layersize[i+1]; 
  conf->nbsynapses *= 2;

  // Allocate the table of synapses:
  conf->synapses = (int*)malloc(2*conf->nbsynapses*sizeof(int));

  // creation of the synapses:
  int j,k=0,l,k2=0,k3=0;
  for(i=1;i<layersize[0];i++) {
    k3 += layersize[i];
    for(j=0; j<layersize[i]; j++) { 
      for(l=0; l<layersize[i+1]; l++) {
        // forward link/synapse:
        conf->synapses[k] = k2+j;
        k++;
        conf->synapses[k] = k3+l;
        k++;
        // Recurrent link/synapse:
        conf->synapses[k] = k3+l;
        k++;
        conf->synapses[k] = k2+j;
        k++;

      }
    }
    k2 += layersize[i];
  }
  return conf;
}

void freeconfig(CONFIG* conf) {
  free(conf->synapses);
  free(conf->layersize);
  free(conf);
}



RNN * creaternn(CONFIG * conf) {

  RNN * net = (RNN*)malloc(sizeof(RNN));
  net->nbneurons = conf->nbneurons;
  net->layersize = (int*)malloc((conf->layersize[0]+1)*sizeof(int));
  int i;
  for(i=0; i<conf->layersize[0]+1; i++) net->layersize[i] = conf->layersize[i];

  // Allocate the neuron table of the Recurrent Neural Network:
  net->n = (NEURON*)malloc(conf->nbneurons*sizeof(NEURON));

  // Initialize some neuron values:
  int j=0,k=0;
  for(i=0; i<conf->nbneurons; i++) {
    if(k==0) { k = conf->layersize[j+1]; j++; }
    net->n[i].layer = j-1;
    net->n[i].nbsynapsesin = 0; 
    k--;
  }

  // Count the incoming synapses for each neuron:
  k=0;
  for(i=0; i<conf->nbsynapses; i++) {
    k++;
    net->n[conf->synapses[k]].nbsynapsesin++;
    k++;
  }

  // Allocate weight table in neurons, and the table of pointer to neuron
  // that represent the incoming synapses:
  for(i=0; i<conf->nbneurons; i++) {
    net->n[i].weight = (double*)malloc(net->n[i].nbsynapsesin*sizeof(double));
    net->n[i].synapsesin = (NEURON**)malloc(net->n[i].nbsynapsesin*sizeof(NEURON*));
    net->n[i].nbsynapsesin = 0;
  }

  // Link the incoming synapses with the neurons:
  k=0;
  for(i=0; i<conf->nbsynapses; i++) {
    k++;
    net->n[conf->synapses[k]].synapsesin[net->n[conf->synapses[k]].nbsynapsesin] = &(net->n[conf->synapses[k-1]]);
    net->n[conf->synapses[k]].nbsynapsesin++;
    k++;
  }

  // Initialization of the values, errors, and weights:
  for(i=0; i<net->nbneurons; i++) {
    for(j=0; j<net->n[i].nbsynapsesin; j++) {
      net->n[i].weight[j] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
    }
    net->n[i].bias = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
    net->n[i].value = 0.0;
    net->n[i].value_prev = 0.0;
    net->n[i].error_prev = 0.0;
    net->n[i].error = 0.0;
  }

  return net;
}


void freernn(RNN * net) {
  int i;
  for(i=0; i<net->nbneurons; i++) {
    free(net->n[i].weight);
    free(net->n[i].synapsesin);
  }
  free(net->n);
  free(net->layersize);
  free(net);
}

void rnnget(RNN * net, double * out) {
  int i,k=0;
  // Store the output of the network in the variable table "out":
  for(i=net->nbneurons-1; i>=(net->nbneurons - net->layersize[net->layersize[0]]); i--) { out[k] = net->n[i].value; k++; }
}

void rnnsetstart(RNN * net, double *input_matrix, double *output_matrix, double *d_weight, int m, int n, int k, size_t size_in_bytes) {
    // Allocate device memory for input and output matrices
    double *d_input, *d_output;
    cudaMalloc(&d_input, size_in_bytes);  // Allocate memory for input matrix
    cudaMalloc(&d_output, size_in_bytes); // Allocate memory for output matrix

    // Copy input matrix to device memory
    cudaMemcpy(d_input, input_matrix, size_in_bytes, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((k + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    matrixMultiplicationKernel<<<gridSize, blockSize>>>(d_input, d_weight, d_output, m, n, k);

    // Copy the result back to host memory
    cudaMemcpy(output_matrix, d_output, size_in_bytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

void rnnset(RNN * net, double * in) {
  int i,j,k;
  double v;

  NEURON *ni,*nj;
  // For each neuron:
  for(i=0; i<net->nbneurons; i++) {
    ni = &(net->n[i]);
    // If it is an input neuron:
    if(i<net->layersize[1]) ni->value = in[i];
    else ni->value = ni->bias;

    // If the neuron is NOT in input layer, then  
    // compute the value from the incoming synapses:
    if(i>=net->layersize[1]) {
      // For each incoming synapse:
      for(j=0; j<ni->nbsynapsesin; j++) {
        nj = ni->synapsesin[j];
        // If the synapse is from input layer to output layer, then tanh the value:
        if(nj->layer == 0 && ni->layer == (net->layersize[0]-1)) {
          ////////////////////////////////////////////////////////////////////////
          // Uncomment the following line to enable reccurent links computation:
          ni->value += tanh(nj->value_prev) * ni->weight[j];
          ////////////////////////////////////////////////////////////////////////
        } else {
          // If it is a forward link/synapse:
          if(ni->layer > nj->layer) ni->value +=  nj->value * ni->weight[j];
          ////////////////////////////////////////////////////////////////////////
          // Uncomment the following line to enable reccurent links computation:
          else ni->value += nj->value_prev * ni->weight[j];
          ////////////////////////////////////////////////////////////////////////
        }
      }
    }
    // If NOT the input layer NOR the output layer, then tanh the value:
    if(ni->layer != 0 && ni->layer != net->layersize[0]-1) ni->value = tanh(ni->value);
  }
}


void rnnlearnstart(RNN * net) {
  int i;
  // For each neuron, initialize error_prev and value_prev for a
  // new training cycle:
  for(i=0; i<net->nbneurons; i++) { net->n[i].error_prev = 0.0; net->n[i].value_prev = 0.0; }
}

void rnnlearn(RNN * net, double * out, double learningrate) {
  int i,j,k;
  k=0;

  NEURON *ni,*nj;
  // Initialize error to zero for the output layer:
  for(i=net->nbneurons-1; i>=net->nbneurons-net->layersize[net->layersize[0]]; i--) net->n[i].error = 0.0;

  // Compute the error for output neurons, and 
  // initialize it to 0 for the other neurons:
  for(i=net->nbneurons-1; i>=0; i--) {
    ni = &(net->n[i]);
    // If ni is an output neuron, update the error:
    if(ni->layer == net->layersize[0]-1) {
      ni->error += ni->value - out[k];
      k++;
    } else {
      ni->error = 0.0;
    }
  }

  // Compute error for all other neurons:
  for(i=net->nbneurons-1; i>=0; i--) {
    ni = &(net->n[i]);
    // For each incoming synapse NOT from output layer:
    for(j=0; j<ni->nbsynapsesin; j++) {
      nj = ni->synapsesin[j];
      // If it is a forward link/synapse:
      if(ni->layer > nj->layer) nj->error += ni->error * ni->weight[j];
    }
  }

  // Update weights:
  for(i=0; i<net->nbneurons; i++) {
    ni = &(net->n[i]);
    double wchange,derivative;
    // For the output layer:
    if(ni->layer == net->layersize[0]-1) {
      derivative = ni->error * learningrate;
      // For each incoming synapse:
      for(j=0; j<ni->nbsynapsesin; j++) {
        nj = ni->synapsesin[j];
        wchange = derivative;
        // If it is a forward link/synapse:
        if(ni->layer > nj->layer) wchange *= nj->value;
        else wchange *= nj->value_prev;
        ni->weight[j] -= wchange;
        if(ni->weight[j] > 5) ni->weight[j] = 5;
        if(ni->weight[j] < -5) ni->weight[j] = -5;
      }
      ni->bias -= derivative;
      if(ni->bias > 5) ni->bias = 5;
      if(ni->bias < -5) ni->bias = -5;

    // For the other layers:
    } else {
      derivative = 1.0 - ni->value * ni->value;
      derivative *= ni->error * learningrate;
      // For each incoming synapse:
      for(j=0; j<ni->nbsynapsesin; j++) {
        nj = ni->synapsesin[j];
        wchange = derivative;
        // If it is a forward link/synapse:
        if(ni->layer > nj->layer) wchange *= nj->value;
        else wchange *= nj->value_prev;
        ni->weight[j] -= wchange;
      }
      ni->bias -= derivative;
    }
  }

  // Update error_prev:
  for(i=0; i<net->nbneurons; i++) net->n[i].error_prev = net->n[i].error;
}

int main() {
    srand(time(NULL));

    // Déclarations des variables pour les matrices et les poids
    float *input_matrix, *output_matrix, *d_weight;
    int k, m, n;
    size_t taille_input_matrix = 16;
    size_t taille_output_matrix = 16;
    size_t taille_d_weight = 16;

    // Allocation de mémoire pour les matrices et les poids sur le GPU
    cudaMalloc((void**)&input_matrix, taille_input_matrix);
    cudaMalloc((void**)&output_matrix, taille_output_matrix);
    cudaMalloc((void**)&d_weight, taille_d_weight);

    int layersize_netrnn[] = { 4, 1, 25, 12, 1 };
    CONFIG * configrnn = createconfig(layersize_netrnn);
    RNN * netrnn = creaternn(configrnn);

    double inc,outc;
    double global_error2 = 1;
    int i2=0;
    int iter;

    //////////////////////////////////////////////////////
    // Training of the Recurrent Neural Network:
    //////////////////////////////////////////////////////

    // Votre boucle d'entraînement ici...

    // Libération de la mémoire allouée sur le GPU
    cudaFree(input_matrix);
    cudaFree(output_matrix);
    cudaFree(d_weight);

    // Libération de la mémoire allouée dynamiquement
    freeconfig(configrnn);
    freernn(netrnn);
    
    return 0;
}
