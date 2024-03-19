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


double mean_squared_error(double *y_true, double *y_pred, int len) {
    double mse = 0.0;
    for (int i = 0; i < len; i++) {
        double error = y_true[i] - y_pred[i];
        mse += error * error;
    }
    return mse / len;
}


double calculate_loss(double **X, double **Y, double **U, double **V, double **W) {

    double loss = 0.0;

    for (int i = 0; i < num_records; i++) { //

        double *x = X[i];
        double *y = Y[i];

        double *prev_activation = (double *)malloc(hidden_dim * sizeof(double));

        for (int j = 0; j < hidden_dim; j++) {
            prev_activation[j] = 0.0;
        }

        for (int timestep = 0; timestep < seq_len; timestep++) {

            double *new_input = (double *)malloc(seq_len * sizeof(double));

            for (int k = 0; k < seq_len; k++) {
                new_input[k] = 0.0;
            }

            new_input[timestep] = x[timestep];


            double *mulu = (double *)malloc(hidden_dim * sizeof(double));
            double *mulw = (double *)malloc(hidden_dim * sizeof(double));
            double _sum = 0.0;

            // Compute mulu
            for (int k = 0; k < hidden_dim; k++) {
                mulu[k] = 0.0;
                for (int l = 0; l < seq_len; l++) {
                    mulu[k] += U[k][l] * new_input[l];
                }
            }

            // Compute mulw
            for (int k = 0; k < hidden_dim; k++) {
                mulw[k] = 0.0;
                for (int l = 0; l < hidden_dim; l++) {
                    mulw[k] += W[k][l] * prev_activation[l];
                }
            }

            // Compute _sum
            for (int k = 0; k < hidden_dim; k++) {
                _sum += mulu[k] + mulw[k];
            }

            
            double activation[hidden_dim];
            for (int k = 0; k < hidden_dim; k++) {
                activation[k] = sigmoid(_sum);
            }

            double *mulv = (double *)malloc(output_dim * sizeof(double));

            // Compute mulv
            for (int k = 0; k < output_dim; k++) {
                mulv[k] = 0.0;
                for (int l = 0; l < hidden_dim; l++) {
                    mulv[k] += V[k][l] * activation[l];
                }
            }

             // Update prev_activation for the next timestep
            for (int k = 0; k < hidden_dim; k++) {
                prev_activation[k] = activation[k];
            }

         // Calculate and add loss per record
        double loss_per_record = (y - mulv[0]) * (y - mulv[0]) / 2.0;
        loss += loss_per_record;

        }
    }

    return loss,activation;
}



typedef struct {
    double *activation;
    double *prev_activation;
} Layer;


Layer *calc_layers(double **x, double **U, double **V, double **W, double *prev_activation, int seq_len, ) {
    
	Layer *layers = (Layer *)malloc(seq_len * sizeof(Layer));
    
    #define HIDDEN_SIZE 2

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

double **train(double **U, double **V, double **W, double **X, double **Y, double **X_validation, double **Y_validation, int max_epochs, double learning_rate, int hidden_dim) {
    

    for (int epoch = 0; epoch < max_epochs; epoch++) {
        // calculate initial loss, ie what the output is given a random set of weights
        double loss = calculate_loss(X, Y, U, V, W);
        double val_loss = calculate_loss(X_validation, Y_validation, U, V, W);

        printf("Epoch: %d, Loss: %f, Validation Loss: %f\n", epoch+1, loss, val_loss);

        // train model/forward pass
        for (int i = 0; i < Y.shape[0]; i++) {
            double **x = X[i];
            double **y = Y[i];

            double *prev_activation = (double *)malloc(hidden_dim * sizeof(double));
            for (int j = 0; j < hidden_dim; j++) {
                prev_activation[j] = 0.0;
            }

            layers, mulu, mulw, mulv = calc_layers(x, U, V, W, prev_activation);

            // difference of the prediction
            double **dmulv = (double **)malloc(hidden_dim * sizeof(double *));

            for (int j = 0; j < hidden_dim; j++) {
                dmulv[j] = (double *)malloc(sizeof(double));
                dmulv[j][0] = mulv[j][0] - y[j][0];
            }

            // Perform backpropagation and get weight updates
            double **dU, **dV, **dW;
            dU, dV, dW = backprop(x, U, V, W, dmulv, mulu, mulw, layers);

            // Update weights
            for (int j = 0; j < hidden_dim; j++) {
                for (int k = 0; k < X.shape[1]; k++) {
                    U[j][k] -= learning_rate * dU[j][k];
                    V[j][k] -= learning_rate * dV[j][k];
                    W[j][k] -= learning_rate * dW[j][k];
                }
            }



           //liberation memoire? utile?

            for (int i = 0; i < num_records - 50; i++) {
                free(X[i]);
                free(Y[i]);
            }
            free(X);
            free(Y);    

            for (int i = 0; i < 50; i++) {
                free(X_validation[i]);
                free(Y_validation[i]);
            }
            free(X_validation);
            free(Y_validation);

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

            for (int j = 0; j < hidden_dim; j++) {
                free(dmulv[j]);
            }
            free(dmulv);


        }
    }


    return U, V, W; // Retourne les poids mis à jour
}





double **backprop(double **x, double **U, double **V, double **W, double *dmulv, double **mulu, double **mulw, Layer *layers, int seq_len, int bptt_truncate) {

    double **dU = (double **)malloc(hidden_dim * sizeof(double *));
    double **dV = (double **)malloc(hidden_dim * sizeof(double *));
    double **dW = (double **)malloc(hidden_dim * sizeof(double *));
    
    double **dU_t = (double **)malloc(hidden_dim * sizeof(double *));
    double **dV_t = (double **)malloc(hidden_dim * sizeof(double *));
    double **dW_t = (double **)malloc(hidden_dim * sizeof(double *));
    
    double **dU_i = (double **)malloc(hidden_dim * sizeof(double *));
    double **dW_i = (double **)malloc(hidden_dim * sizeof(double *));
    
    double _sum;
    double *dsv;

    for (int i = 0; i < hidden_dim; i++) {
        dU[i] = (double *)malloc(seq_len * sizeof(double));
        dW[i] = (double *)malloc(hidden_dim * sizeof(double));
        
        dU_t[i] = (double *)malloc(seq_len * sizeof(double));
        dW_t[i] = (double *)malloc(hidden_dim * sizeof(double));
        
        dU_i[i] = (double *)malloc(seq_len * sizeof(double));
        dW_i[i] = (double *)malloc(hidden_dim * sizeof(double));
    }
   
    *dV = (double *)malloc(hidden_dim * sizeof(double));
    *dV_t = (double *)malloc(hidden_dim * sizeof(double));

    for (int i = 0; i < hidden_dim; i++) {
        for (int j = 0; j < seq_len; j++) {
            dU[i][j] = 0.0;
            dU_t[i][j] = 0.0;
            dU_i[i][j] = 0.0;
        }
        for (int j = 0; j < hidden_dim; j++) {
            dW[i][j] = 0.0;
            dW_t[i][j] = 0.0;
            dW_i[i][j] = 0.0;
        }
    }
    **dV = 0.0;
     **dV_t = 0.0;

    // Calculation
    dsv = (double *)malloc(hidden_dim * sizeof(double));
    _sum = **mulu + **mulw;
    for (int timestep = 0; timestep < seq_len; timestep++) {
        *dV_t = 0.0;
        for (int i = 0; i < hidden_dim; i++) {
            dV_t[0][i] = 0.0;
        }
        dV_t = matmul(dmulv, transpose(layers[timestep].activation)); // Assuming matmul() is a function that performs matrix multiplication
        for (int i = 0; i < hidden_dim; i++) {
            dsv[i] = dmulv[0] * layers[timestep].activation[i];
        }
        double *dprev_activation = get_previous_activation_differential(_sum, dsv, W);

        for (int k = timestep - 1; k >= fmax(-1, timestep - bptt_truncate - 1); k--) {
            for (int i = 0; i < hidden_dim; i++) {
                dsv[i] += dprev_activation[i];
            }
            dprev_activation = get_previous_activation_differential(_sum, dsv, W);
            dW_i = matmul(W, layers[timestep].prev_activation); // Assuming matmul() is a function that performs matrix multiplication

            double *new_input = (double *)malloc(seq_len * sizeof(double));
            for (int i = 0; i < seq_len; i++) {
                new_input[i] = 0.0;
            }
            new_input[timestep] = x[timestep];
            dU_i = matmul(U, new_input); // Assuming matmul() is a function that performs matrix multiplication

            for (int i = 0; i < hidden_dim; i++) {
                for (int j = 0; j < seq_len; j++) {
                    dU_t[i][j] += dU_i[i][j];
                    dW_t[i][j] += dW_i[i][j];
                }
            }
        }
        for (int i = 0; i < hidden_dim; i++) {
            for (int j = 0; j < seq_len; j++) {
                dU[i][j] += dU_t[i][j];
                dW[i][j] += dW_t[i][j];
            }
        }
    }

    // Clipping gradients
    for (int i = 0; i < hidden_dim; i++) {
        for (int j = 0; j < seq_len; j++) {
            if (dU[i][j] > max_clip_val) {
                dU[i][j] = max_clip_val;
            }
            if (dU[i][j] < min_clip_val) {
                dU[i][j] = min_clip_val;
            }
        }
        for (int j = 0; j < hidden_dim; j++) {
            if (dW[i][j] > max_clip_val) {
                dW[i][j] = max_clip_val;
            }
            if (dW[i][j] < min_clip_val) {
                dW[i][j] = min_clip_val;
            }
        }
    }
    if (**dV > max_clip_val) {
        **dV = max_clip_val;
    }
    if (**dV < min_clip_val) {
        **dV = min_clip_val;
    }

    return dU, dV, dW;
}

double *get_previous_activation_differential(double _sum, double *ds, double **W) {
    double *d_sum = (double *)malloc(hidden_dim * sizeof(double));
    for (int i = 0; i < hidden_dim; i++) {
        d_sum[i] = _sum * (1 - _sum) * ds[i];
    }
    double *dmulw = (double *)malloc(hidden_dim * sizeof(double));
    for (int i = 0; i < hidden_dim; i++) {
        dmulw[i] = d_sum[i] * 1.0; // Ici, l'opération `np.ones_like(ds)` en Python est remplacée par 1.0 en C, car il



















int main () {

    double sin_wave[200];
    double **X, **Y, **X_validation, **Y_validation;

    // Generate sin wave data
    for (int i = 0; i < 200; i++) {
        sin_wave[i] = sin(i);
    }

    // Allocate memory for training data
    int num_records = 200 - seq_len;

    X = (double **)malloc(num_records * sizeof(double *));
    Y = (double **)malloc(num_records * sizeof(double *));

    for (int i = 0; i < num_records - 50; i++) {

        X[i] = (double *)malloc(seq_len * sizeof(double));
        Y[i] = (double *)malloc(sizeof(double));

        for (int j = 0; j < seq_len; j++) {
            X[i][j] = sin_wave[i + j];
        }
        Y[i][0] = sin_wave[i + seq_len];
    }

    // Allocate memory for validation data
    X_validation = (double **)malloc(50 * sizeof(double *));
    Y_validation = (double **)malloc(50 * sizeof(double *));

    for (int i = num_records - seq_len; i < num_records; i++) {
        X_validation[i - num_records + seq_len] = (double *)malloc(seq_len * sizeof(double));
        Y_validation[i - num_records + seq_len] = (double *)malloc(sizeof(double));

        for (int j = 0; j < seq_len; j++) {
            X_validation[i - num_records + seq_len][j] = sin_wave[i + j];
        }
        Y_validation[i - num_records + seq_len][0] = sin_wave[i + seq_len];
    }

    double **U, **V, **W;

    // Allocate memory for weights
    U = (double **)malloc(hidden_dim * sizeof(double *));
    V = (double **)malloc(output_dim * sizeof(double *));
    W = (double **)malloc(hidden_dim * sizeof(double *));

    for (int i = 0; i < hidden_dim; i++) {
        U[i] = (double *)malloc(seq_len * sizeof(double));
        W[i] = (double *)malloc(hidden_dim * sizeof(double));
    }
    for (int i = 0; i < output_dim; i++) {
        V[i] = (double *)malloc(hidden_dim * sizeof(double));
    }

    // Initialize weights randomly
    srand(12161);
    for (int i = 0; i < hidden_dim; i++) {
        for (int j = 0; j < seq_len; j++) {
            U[i][j] = (double)rand() / RAND_MAX;
        }
    }
    for (int i = 0; i < output_dim; i++) {
        for (int j = 0; j < hidden_dim; j++) {
            V[i][j] = (double)rand() / RAND_MAX;
        }
    }
    for (int i = 0; i < hidden_dim; i++) {
        for (int j = 0; j < hidden_dim; j++) {
            W[i][j] = (double)rand() / RAND_MAX;
        }
    }

    // Train the RNN
    U, V, W = train(U, V, W, X, Y, X_validation, Y_validation);

    return 0;
};
