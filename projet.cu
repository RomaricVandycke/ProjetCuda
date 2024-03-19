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

	
#define HIDDEN_SIZE 2

Layer *calc_layers(double **x, double **U, double **V, double **W, double *prev_activation, int seq_len, ) {
    
	Layer *layers = (Layer *)malloc(seq_len * sizeof(Layer));
    
	double mulu[HIDDEN_SIZE][input_size];
    double mulw[HIDDEN_SIZE][HIDDEN_SIZE];
    double mulv[HIDDEN_SIZE][HIDDEN_SIZE];

    for (int timestep = 0; timestep < seq_len; timestep++) {
        double new_input[input_size];
        for (int i = 0; i < input_size; i++) {
            new_input[i] = 0;
        }
        new_input[timestep] = x[timestep][0]; 

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

    return layers, mulu, mulw, mulv;
}

int main () {

    // creation des matrices
    int seq_len = 3;
    int input_size = 2;
    double **x = malloc(seq_len * sizeof(double *));
    double **U = malloc(HIDDEN_SIZE * sizeof(double *));
    double **V = malloc(HIDDEN_SIZE * sizeof(double *));
    double **W = malloc(HIDDEN_SIZE * sizeof(double *));
    double *prev_activation = malloc(HIDDEN_SIZE * sizeof(double));

    

    Layer *result = calc_layers(x, U, V, W, prev_activation, seq_len, input_size);

    // suite script

    // free memory
    for (int i = 0; i < seq_len; i++) {
        free(result[i].activation);
        free(result[i].prev_activation);
    }
    free(result);
    free(prev_activation);
    for (int i = 0; i < seq_len; i++) {
        free(x[i]);
    }
    free(x);

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        free(U[i]);
        free(V[i]);
        free(W[i]);
    }
    free(U);
    free(V);
    free(W);

    return 0;
};

double train(double **U,) {
    for (int epoch = O; i < max_epochs. i++) {

        double *loss, prev_activation = calculate_loss(X, Y, U, V, W)

    }
}

def train(U, V, W, X, Y, X_validation, Y_validation):
    for epoch in range(max_epochs):
        # calculate initial loss, ie what the output is given a random set of weights
        loss, prev_activation = calculate_loss(X, Y, U, V, W)

        # check validation loss
        val_loss, _ = calculate_loss(X_validation, Y_validation, U, V, W)

        print(f'Epoch: {epoch+1}, Loss: {loss}, Validation Loss: {val_loss}')

        # train model/forward pass
        for i in range(Y.shape[0]):
            x, y = X[i], Y[i]
            layers = []
            prev_activation = np.zeros((hidden_dim, 1))

            layers, mulu, mulw, mulv = calc_layers(x, U, V, W, prev_activation)

            # difference of the prediction
            dmulv = mulv - y
            dU, dV, dW = backprop(x, U, V, W, dmulv, mulu, mulw, layers)

            # update weights
            U -= learning_rate * dU
            V -= learning_rate * dV
            W -= learning_rate * dW

    return U, V, W

