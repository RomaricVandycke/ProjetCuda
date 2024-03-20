typedef double mlp_float;

typedef struct {
    mlp_float *synaptic_weight;
    mlp_float *neuron_value;
    mlp_float *neuron_error_value;
    mlp_float *input_neuron;
    mlp_float *output_neuron;
    mlp_float *output_error_value;
    int *layer_index;
    int *layer_size;
    int *synapse_index;
    int layer_number;
    int neuron_number;
    int synapse_number;
    int input_layer_size;
    int output_layer_size;
} mlp;

static mlp_float MAGICAL_WEIGHT_NUMBER = 1.0f;
static mlp_float MAGICAL_LEARNING_NUMBER = 0.4f;

void reinit_mlp(mlp * network) {
int i;
for (i = 0; i < network->synapse_number; i++) {
        network->synaptic_weight[i] = /*0.001;*/MAGICAL_WEIGHT_NUMBER * (mlp_float)rand() / RAND_MAX - MAGICAL_WEIGHT_NUMBER/2;
}
}

mlp *create_mlp(int layer_number, int *layer_size) {

    mlp *network = (mlp*)malloc(sizeof * network);

    network->layer_number = layer_number;
    network->layer_size = (int*)malloc(sizeof * network->layer_size * network->layer_number);
    network->layer_index = (int*)malloc(sizeof * network->layer_index * network->layer_number);

    int i;
    network->neuron_number = 0;
    for (i = 0; i < layer_number; i++) {
        network->layer_size[i] = layer_size[i];
        network->layer_index[i] = network->neuron_number;
        network->neuron_number += layer_size[i];
    }

    network->neuron_value = (mlp_float*)malloc(sizeof * network->neuron_value * network->neuron_number);
    network->neuron_error_value = (mlp_float*)malloc(sizeof * network->neuron_error_value * network->neuron_number);

    network->input_layer_size = layer_size[0];
    network->output_layer_size = layer_size[layer_number-1];
    network->input_neuron = network->neuron_value;
    network->output_neuron = &network->neuron_value[network->layer_index[layer_number-1]];
    network->output_error_value = &network->neuron_error_value[network->layer_index[layer_number-1]];

    network->synapse_index = (int*)malloc(sizeof * network->synapse_index * (network->layer_number-1));
    network->synapse_number = 0;
    for (i = 0; i < layer_number - 1; i++) {
        network->synapse_index[i] = network->synapse_number;
        network->synapse_number += (network->layer_size[i]+1) * network->layer_size[i+1];
    }

    network->synaptic_weight = (mlp_float*)malloc(sizeof * network->synaptic_weight * network->synapse_number);


    for (i = 0; i < network->synapse_number; i++) {
        network->synaptic_weight[i] = MAGICAL_WEIGHT_NUMBER * (mlp_float)rand() / RAND_MAX - MAGICAL_WEIGHT_NUMBER/2;
    }
    return network;
}

void free_mlp (mlp *network) {
    free(network->layer_size);
    free(network->layer_index);
    free(network->neuron_value);
    free(network->neuron_error_value);
    free(network->synapse_index);
    free(network->synaptic_weight);
    free(network);
}

void set_mlp (mlp * network, mlp_float *vector) {
    if (vector != NULL) {
        int i;
        for (i = 0; i < network->input_layer_size; i++) {
            network->input_neuron[i] = vector[i];
        }
    }
    int i;
    int synapse_index;
    synapse_index = 0;
    for (i = 1; i < network->layer_number; i++) {
        int j;
        for (j = network->layer_index[i]; j < network->layer_index[i] + network->layer_size[i]; j++) {
            mlp_float weighted_sum = 0.0;
            int k;
            for (k = network->layer_index[i-1]; k < network->layer_index[i-1] + network->layer_size[i-1]; k++) {
                weighted_sum += network->neuron_value[k] * network->synaptic_weight[synapse_index];
                synapse_index++;
            }
            weighted_sum += network->synaptic_weight[synapse_index];

            synapse_index++;
            network->neuron_value[j] = weighted_sum;
            if (i != network->layer_number - 1) network->neuron_value[j] = tanh(network->neuron_value[j]);
        }
    }
}

void get_mlp (mlp *network, mlp_float *vector) {
    int i;
    for (i = 0; i < network->output_layer_size; i++) {
        vector[i] = network->output_neuron[i];
    }
}

void learn_mlp (mlp *network, mlp_float *desired_out, mlp_float learning_rate) {

    int i;
    mlp_float global_error = 0;
    int synapse_index = network->synapse_index[network->layer_number-2];

    for (i = 0; i < network->output_layer_size; i++) {
        network->output_error_value[i] = network->output_neuron[i] - desired_out[i];
        int j;
        for (j = network->layer_index[network->layer_number-2]; j < network->layer_index[network->layer_number-2] + network->layer_size[network->layer_number-2]; j++) {
            mlp_float weightChange;
            weightChange = learning_rate * network->output_error_value[i] * network->neuron_value[j];
            network->synaptic_weight[synapse_index] -= weightChange;

            if (network->synaptic_weight[synapse_index] > 5) network->synaptic_weight[synapse_index] = 5;
            if (network->synaptic_weight[synapse_index] < -5) network->synaptic_weight[synapse_index] = -5;

            synapse_index++;
        }
        mlp_float weightChange;
        weightChange = learning_rate * network->output_error_value[i];
        network->synaptic_weight[synapse_index] -= weightChange;

        if (network->synaptic_weight[synapse_index] > 5) network->synaptic_weight[synapse_index] = 5;
        if (network->synaptic_weight[synapse_index] < -5) network->synaptic_weight[synapse_index] = -5;

        synapse_index++;
    }

    for (i = network->layer_number - 2; i > 0; i--) {
        int j;
        int jj= 0;
        int synapse_index = network->synapse_index[i-1];
        for (j = network->layer_index[i]; j < network->layer_index[i] + network->layer_size[i]; j++,jj++) {
            int k;

            int synapse_index2 = network->synapse_index[i] + jj;
            network->neuron_error_value[j] = 0;
            for (k = network->layer_index[i+1]; k < network->layer_index[i+1] + network->layer_size[i+1]; k++) {
                network->neuron_error_value[j] += network->synaptic_weight[synapse_index2] * network->neuron_error_value[k];
                synapse_index2+=network->layer_size[i]+1;
            }

            for (k = network->layer_index[i-1]; k < network->layer_index[i-1] + network->layer_size[i-1]; k++) {

                mlp_float weightChange;

                weightChange = 1.0 - network->neuron_value[j] * network->neuron_value[j];
                weightChange *= network->neuron_error_value[j] * learning_rate;
                weightChange *= network->neuron_value[k];

                network->synaptic_weight[synapse_index] -= weightChange;
                synapse_index++;
            }
            mlp_float weightChange;

            weightChange = 1.0 - network->neuron_value[j] * network->neuron_value[j];
            weightChange *= network->neuron_error_value[j] * learning_rate;
            
            network->synaptic_weight[synapse_index] -= weightChange;

            synapse_index++;
            
            
        }
    }
}




void get_mlp_inputs (mlp *network, mlp_float *vector) {
    if (vector != NULL) {
        int i;
        for (i = 0; i < network->input_layer_size; i++) {
            vector[i] = network->input_neuron[i];
        }
    }
}
