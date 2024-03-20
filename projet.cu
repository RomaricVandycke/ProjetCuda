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
	return 1/(1+exp(-x))
};


double mean_squared_error(double *y_true, double *y_pred, int len) {
    double mse = 0.0;
    for (int i = 0; i < len; i++) {
        double error = y_true[i] - y_pred[i];
        mse += error * error;   
    }
    return mse / len;
}


double calculate_loss(double **X, double **Y, double **U, double **V, double **W, double *loss_, double *activation_) {

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

            double _sum = 0.0;

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
 
    *loss_ = loss;
    *activation_ = activation;


    return 0;
}



typedef struct {
    double *activation;
    double *prev_activation;
} Layer;


Layer *calc_layers(double **x, double **U, double **V, double **W, double *prev_activation, double &mulu, double &mulw, double &mulv) {
    
	Layer *layers = (Layer *)malloc(seq_len * sizeof(Layer));
    
    double *mulu = (double *)malloc(hidden_dim * sizeof(double));
    double *mulv = (double *)malloc(output_dim * sizeof(double));
    double *mulw = (double *)malloc(hidden_dim * sizeof(double));

    for (int timestep = 0; timestep < seq_len; timestep++) {

        double *new_input = (double *)malloc(seq_len * sizeof(double));
        for (int k = 0; k < seq_len; k++) {
            new_input[k] = 0.0;
        }

        new_input[timestep] = x[timestep];


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

        double _sum = 0.0;

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


        layers[timestep].activation = (double *)malloc(hidden_dim * sizeof(double));
        layers[timestep].prev_activation = (double *)malloc(hidden_dim * sizeof(double));
        
        for (int i = 0; i < hidden_dim; i++) {
            layers[timestep].activation[i] = activation[i];
            layers[timestep].prev_activation[i] = prev_activation[i];
        }

        // Update prev_activation for the next timestep
        for (int k = 0; k < hidden_dim; k++) {
            prev_activation[k] = activation[k];
        }

    }
    
     &mulu = mulu;
     &mulw = mulw;
     &mulv = mulv;

    //pb : une fct ne renvoie qun param enC
    return layers;
}







double **backprop(double **x, double **U, double **V, double **W, double *dmulv, double **mulu, double **mulw, Layer *layers, double &dU, double &dV, double &dW) {



    double **dU = (double **)malloc(hidden_dim * sizeof(double *));
    double **dV = (double **)malloc(output_dim * sizeof(double *));
    double **dW = (double **)malloc(hidden_dim * sizeof(double *));
    
    double **dU_t = (double **)malloc(hidden_dim * sizeof(double *));
    double **dV_t = (double **)malloc(output_dim * sizeof(double *));
    double **dW_t = (double **)malloc(hidden_dim * sizeof(double *)); 
    
    double **dU_i = (double **)malloc(hidden_dim * sizeof(double *));
    double **dW_i = (double **)malloc(hidden_dim * sizeof(double *));


    for (int i = 0; i < hidden_dim; i++) {
        dU[i] = (double *)malloc(seq_len * sizeof(double));
        dW[i] = (double *)malloc(hidden_dim * sizeof(double));
        
        dU_t[i] = (double *)malloc(seq_len * sizeof(double));
        dW_t[i] = (double *)malloc(hidden_dim * sizeof(double));
        
        dU_i[i] = (double *)malloc(seq_len * sizeof(double));
        dW_i[i] = (double *)malloc(hidden_dim * sizeof(double));
    }
    
    for (int i = 0; i < output_dim; i++) {
        dV[i] = (double *)malloc(hidden_dim * sizeof(double));
        dV_t[i] = (double *)malloc(hidden_dim * sizeof(double));
    }

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

    for (int i = 0; i < output_dim; i++) {
        for (int j = 0; j < hidden_dim; j++) {
            dV[i][j] = 0.0;
            dV_t[i][j] = 0.0;
        }
    }



    // Calculation


    double _sum;
    _sum = **mulu + **mulw;

    double **dsv = (double **)malloc(hidden_dim * sizeof(double *)); 
    // Produit matriciel entre la transposée de W et dmulw
    for (int i = 0; i < hidden_dim; i++) {
        dsv[i] = (double *)malloc(hidden_dim * sizeof(double));
        for (int j = 0; j < hidden_dim; j++) {
            dsv[i][j] = 0;
            for (int k = 0; k < hidden_dim; k++) {
                dsv[i][j] += V[k][i] * dmulv[k];
            }
        }
    }


    double *get_previous_activation_differential(double _sum, double *ds, double **W) {
        
        double *d_sum = (double *)malloc(hidden_dim * sizeof(double));
        
        for (int i = 0; i < hidden_dim; i++) {
            d_sum[i] = _sum * (1 - _sum) * ds[i];
        }


        double *dmulw = (double *)malloc(hidden_dim * sizeof(double));
        
        for (int i = 0; i < hidden_dim; i++) {
            dmulw[i] = d_sum[i] * 1.0; // Ici, l'opération `np.ones_like(ds)` en Python est remplacée par 1.0 en C, car il
        }


        double **result = (double **)malloc(hidden_dim * sizeof(double *)); 
        // Produit matriciel entre la transposée de W et dmulw
        for (int i = 0; i < hidden_dim; i++) {
            result[i] = (double *)malloc(hidden_dim * sizeof(double));
            for (int j = 0; j < hidden_dim; j++) {
                result[i][j] = 0;
                for (int k = 0; k < hidden_dim; k++) {
                    result[i][j] += W[k][i] * dmulw[k];
                }
            }
        }

        return result;
    }



    for (int timestep = 0; timestep < seq_len; timestep++) {
        
        *dV_t = 0.0;
        
        for (int i = 0; i < hidden_dim; i++) {
            dV_t[0][i] = 0.0;
        }
        dV_t = matmul(dmulv, transpose(layers[timestep].activation)); // Assuming matmul() is a function that performs matrix multiplication
        
        double ds = dsv;

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

    &dU = dU
    &dV = dV
    &dW = dW

    return 0;
}





double **train(double **U, double **V, double **W, double **X, double **Y, double **X_validation, double **Y_validation) {
    

    for (int epoch = 0; epoch < max_epochs; epoch++) {
        // calculate initial loss, ie what the output is given a random set of weights
        //double loss, double val_loss  = calculate_loss(X, Y, U, V, W);
        //double val_loss = calculate_loss(X_validation, Y_validation, U, V, W);
        double loss_training,preactivation_training;
        calculate_loss(X, Y, U, V, W, &loss_training, &preactivation_training);
        
        double loss_validation , _ ;
        calculate_loss(X_validation, Y_validation, U, V, W, &loss_validation, &_);
        //double loss, val_loss;
        //loss, val_loss = calculate_loss(X, Y, U, V, W);


        printf("Epoch: %d, Loss: %f, Validation Loss: %f\n", epoch+1, loss, val_loss);

        // train model/forward pass
        for (int i = 0; i < Y.shape[0]; i++) {
            double **x = X[i];
            double **y = Y[i];

            double *prev_activation = (double *)malloc(hidden_dim * sizeof(double));
            for (int j = 0; j < hidden_dim; j++) {
                prev_activation[j] = 0.0;
            }
            
            double mulu, mulw, mulv ;
            layers = calc_layers(x, U, V, W, prev_activation, &mulu,&mulw, &mulv);

            // difference of the prediction
            double **dmulv = (double **)malloc(hidden_dim * sizeof(double *));

            for (int j = 0; j < hidden_dim; j++) {
                dmulv[j] = (double *)malloc(sizeof(double));
                dmulv[j][0] = mulv[j][0] - y[j][0];
            }

            // Perform backpropagation and get weight updates
            double dU, dV, dW;
            backprop(x, U, V, W, dmulv, mulu, mulw, layers, &dU, &dV, &dW);

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

            for (int i = 0; i < hidden_dim; i++) {
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

    return 0; // Retourne les poids mis à jour
}








int main () {

    double sin_wave[200];

    for (int i = 0; i < 200; i++) {
        sin_wave[i] = sin(i);
    }

    // Allocate memory for training data
    int num_records = 200 - seq_len;
    double **X, **Y;

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

    double  **X_validation, **Y_validation;
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


    srand(12161);
  
    double **U;
    U = (double **)malloc(hidden_dim * sizeof(double *));
    for (int i = 0; i < hidden_dim; i++) {
        U[i] = (double *)malloc(seq_len * sizeof(double));
        for (int j = 0; j < seq_len; j++) {
            U[i][j] = (double)rand() / RAND_MAX;
        }
    }
    
    double **V;
    V = (double **)malloc(output_dim * sizeof(double *));
    for (int i = 0; i < output_dim; i++) {
        V[i] = (double *)malloc(hidden_dim * sizeof(double));
        for (int j = 0; j < hidden_dim; j++) {
            V[i][j] = (double)rand() / RAND_MAX;
        }
    }

    double **W;
    W = (double **)malloc(hidden_dim * sizeof(double *));
    for (int i = 0; i < hidden_dim; i++) {
        W[i] = (double *)malloc(hidden_dim * sizeof(double));
        for (int j = 0; j < hidden_dim; j++) {
            W[i][j] = (double)rand() / RAND_MAX;
        }
    }

    // Train the RNN
    
    train(U, V, W, X, Y, X_validation, Y_validation);

    return 0;
};
