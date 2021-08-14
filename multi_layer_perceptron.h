#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>
#include<string.h>

double Momentum = 0.9 ;
double Rate = 0.05 ;
double Epsilon = 0.000000001 ;
int act_choice ;
int test_index ;

typedef struct
{
    int size ;
    double *weights, *prev_weights ;
} Neuron ;

typedef struct
{
    int num_neurons ;
    Neuron *arr_neuron ;
} Layer ;

typedef struct
{
    int num_layers, num_inputs ;
    Layer *arr_layer ;
} Network ;


double * layer_output(Network *mlp,double *input_data_single,int layer,int act) ;
double avg_quad_err(Network *mlp,double **input_data,double **output_data,int num_data) ;
double * delta_calc(Network *mlp, int layer, double *input_data_single, double *output_data_single) ;
void grad_desc_momentum(Network *mlp, int layer, double *input_data_single, double *output_data_single, double *delta, double *prev_layer_act) ;
void grad_desc_regular(Network *mlp, int layer, double *input_data_single, double *output_data_single, double *delta, double *prev_layer_act) ;
double activation(double x) ;
double deriv_act(double x) ;
void act_softmax(double *output, int num_out) ;
void print_acc(Network *mlp, double **input_data, double **output_data, int num_data, int index) ;
void print_weights(Network *mlp) ;