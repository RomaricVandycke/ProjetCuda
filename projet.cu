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

int main () {

};
