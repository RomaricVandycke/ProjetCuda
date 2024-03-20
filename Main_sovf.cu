#include <stdio.h>
#include <time.h>
#include <math.h>
#include <malloc.h>
#include <stdlib.h>

#include "mlp.h"


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

void rnnsetstart(RNN * net) {
  int i,j;

  NEURON *ni,*nj;
  // For each neuron, update value_prev:
  for(i=0; i<net->nbneurons; i++) {
    ni = &(net->n[i]);
    // If NOT the output layer, then the value is already computed by tanh:
    if(ni->layer != net->layersize[0]-1) ni->value_prev = ni->value;
    else ni->value_prev = tanh(ni->value);
  }
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
  int layersize[] = {1, 25, 12, 1};
  int layersize_netrnn[] = { 4, 1, 25, 12, 1 };

  mlp * netmlp = create_mlp (4, layersize);
  srand(time(NULL));
  CONFIG * configrnn = createconfig(layersize_netrnn);
  RNN * netrnn = creaternn(configrnn);

  double inc,outc;

  double global_error = 1;
  double global_error2 = 1;

  int iter,i1=0,i2=0;


  //////////////////////////////////////////////////////
  // Training of the Multi-Layer Perceptron:
  //////////////////////////////////////////////////////
  while(global_error > 0.005 && i1<1000) {

    for (iter=0; iter < 100; iter++) {
      inc = 1.0*rand()/(RAND_MAX+1.0);
      outc = inc*inc;
      set_mlp(netmlp,&inc);
      learn_mlp(netmlp,&outc,0.03);
    }

    global_error = 0;
    int k;
    for (k=0; k < 100; k++) {
      inc = 1.0*rand()/(RAND_MAX+1.0);
      outc = inc*inc;
      set_mlp(netmlp,&inc);
      get_mlp(netmlp,&outc);
      mlp_float desired_out = inc*inc;
      global_error += (desired_out - outc)*(desired_out - outc);
    }         
    global_error /= 100;
    global_error = sqrt(global_error);

    i1++;
  }

  //////////////////////////////////////////////////////
  // Training of the Recurrent Neural Network:
  //////////////////////////////////////////////////////
  while(global_error2 > 0.005 && i2<1000) {

    rnnlearnstart(netrnn);

    for (iter=0; iter < 100; iter++) {
      inc = 1.0*rand()/(RAND_MAX+1.0);
      outc = inc*inc;
      rnnsetstart(netrnn);
      rnnset(netrnn,&inc);
      double outc2;

      rnnlearn(netrnn,&outc,0.03);
    }

    global_error2 = 0;
    int k;
    for (k=0; k < 100; k++) {
      inc = 1.0*rand()/(RAND_MAX+1.0);
      outc = inc*inc;
      double desired_out = inc*inc;

      rnnsetstart(netrnn);
      rnnset(netrnn,&inc);
      rnnget(netrnn,&outc);

      global_error2 += (desired_out - outc)*(desired_out - outc);
    }
    global_error2 /= 100;
    global_error2 = sqrt(global_error2);
    if(!isnormal(global_error2)) global_error2 = 100;
    i2++;
  }

  //////////////////////////////////////////////////////
  // Test of performance for the both networks:
  //////////////////////////////////////////////////////
  global_error = 0;
  global_error2 = 0;

  int k;
  for (k=0; k < 10000; k++) {
    inc = 1.0*rand()/(RAND_MAX+1.0);
    outc = inc*inc;
    double desired_out = inc*inc;

    rnnsetstart(netrnn);
    rnnset(netrnn,&inc);
    rnnget(netrnn,&outc);
    global_error2 += (desired_out - outc)*(desired_out - outc);

    set_mlp(netmlp,&inc);
    get_mlp(netmlp,&outc);
    global_error += (desired_out - outc)*(desired_out - outc);
  }

  global_error /= 10000;
  global_error = sqrt(global_error);
  printf("\n  MLP:    Training cycles: %5d    Error: %f",i1,global_error);
  global_error2 /= 10000;
  global_error2 = sqrt(global_error2);
  printf("\n  RNN:    Training cycles: %5d    Error: %f",i2,global_error2);

  free_mlp(netmlp);
  freeconfig(configrnn);
  freernn(netrnn);

}
