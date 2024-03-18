#include <math.h>
#include <stdio.h>
#include <stdlib.h>

float learning_rate = 0.0001;
int seq_len = 50;
int max_epochs = 25;
int hidden_dim = 100;
int output_dim = 1;
int bptt_truncate = 5;
int min_clip_val = -10;
int max_clip_val = 10;

double sigmoid(float x){
	return 1/(1+exp(-x)
};

double calculate_loss(double** X, double** Y, double** U, double** V, double** W) {
	float loss = 0;
	int taille_Y = ;
	for (i=0; i<taille_Y;i++) {
		
};


typedef struct {
    double *activation;
    double *prev_activation;
} Layer;

	

Layer *calc_layers(double **x, double **U, double **V, double **W, double *prev_activation, int seq_len, int input_size) {
    Layer *layers = (Layer *)malloc(seq_len * sizeof(Layer));
    double mulu[HIDDEN_SIZE][input_size];
    double mulw[HIDDEN_SIZE][HIDDEN_SIZE];
    double mulv[HIDDEN_SIZE][HIDDEN_SIZE];

    for (int timestep = 0; timestep < seq_len; timestep++) {
        double new_input[input_size];
        for (int i = 0; i < input_size; i++) {
            new_input[i] = 0;
        }
        new_input[timestep] = x[timestep][0]; // Assuming x is a 2D array

        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < input_size; j++) {
                mulu[i][j] = 0;
                for (int k = 0; k < input_size; k++) {
                    mulu[i][j] += U[i][k] * new_input[k];
                }
            }
        }

        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                mulw[i][j] = 0;
                for (int k = 0; k < HIDDEN_SIZE; k++) {
                    mulw[i][j] += W[i][k] * prev_activation[k];
                }
            }
        }

        double sum[HIDDEN_SIZE];
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            sum[i] = 0;
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                sum[i] += mulw[i][j] + mulu[i][j];
            }
        }

        double activation[HIDDEN_SIZE];
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            activation[i] = sigmoid(sum[i]);
        }

        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                mulv[i][j] = 0;
                for (int k = 0; k < HIDDEN_SIZE; k++) {
                    mulv[i][j] += V[i][k] * activation[k];
                }
            }
        }

        layers[timestep].activation = (double *)malloc(HIDDEN_SIZE * sizeof(double));
        layers[timestep].prev_activation = (double *)malloc(HIDDEN_SIZE * sizeof(double));
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            layers[timestep].activation[i] = activation[i];
            layers[timestep].prev_activation[i] = prev_activation[i];
        }

        for (int i = 0; i < HIDDEN_SIZE; i++) {
            prev_activation[i] = activation[i];
        }
    }

    return layers;
}

int main () {

};
