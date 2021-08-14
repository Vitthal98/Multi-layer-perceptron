#include "multi_layer_perceptron.h"

int main(int argc , char **argv)
{
    time_t seed = 0 ;
    time(&seed) ;
    srand(seed) ;

    int data_set ;
    printf("\nEnter the choice of Dataset to be tested: \n\t1: Blood Transfusion Binary Classification \n\t2: Contraceptive Multi-Classification \n") ;
    scanf("%d",&data_set) ;

    double **input_data,**output_data ;
    int num_data ;
    FILE *fptr ;
    int num_inputs , num_layers ;
    int *num_neurons ;
    int count = 0 ;
    if(data_set == 1)
    {
        fptr = fopen("transfusion1.txt" ,"r") ;
        for(char ch = fgetc(fptr); ch != EOF ; ch = fgetc(fptr))
        {
            if(ch=='\n')
            count++ ;
        }
        fclose(fptr) ;
        
        count-- ;
        input_data = (double **)malloc(sizeof(double *) * count) ;
        output_data = (double **)malloc(sizeof(double *) * count) ;
        num_data = 500 ;
        num_inputs = 4 ;

        fptr = fopen("transfusion1.txt" ,"r") ;
        if(fptr != NULL)
        {
            for(int i = 0 ; i < count ; i++)
            {
                input_data[i] = (double *)malloc(sizeof(double) * 3) ;
                output_data[i] = (double *)malloc(sizeof(double) * 2) ;
                double temp,red ;
                fscanf(fptr, "%lf ,%lf,%lf,%lf ,%lf\n", &(input_data[i][0]), &(input_data[i][1]), &red, &(input_data[i][2]), &temp) ;
                if(temp == 0)
                {
                    output_data[i][0] = 1 ;
                    output_data[i][1] = 0 ;
                }    
                else
                {
                    output_data[i][1] = 1 ;
                    output_data[i][0] = 0 ;
                }    
                    
            }    
        }
        fclose(fptr) ;

        double *max_arr = (double *)malloc(sizeof(double) * 3) ;
        for(int i = 0 ; i < 3 ; i++)
            max_arr[i] = 0.0 ;
        for(int i = 0 ; i < count ; i++)
        {
            for(int j = 0 ; j < 3 ; j++ )
            {
                if(max_arr[j] < input_data[i][j])
                    max_arr[j] = input_data[i][j] ;
            }
        }
        for(int i = 0 ; i < count ; i++)
        {
            for(int j = 0 ; j < 3 ; j++ )
            {
                input_data[i][j] = input_data[i][j] / (double)max_arr[j] ;
                input_data[i][j] -= 1 ;
            }
        }
        
    }
    else if(data_set == 2)
    {
        fptr = fopen("cmc.txt" ,"r") ;
        for(char ch = fgetc(fptr); ch != EOF ; ch = fgetc(fptr))
        {
            if(ch=='\n')
            count++ ;
        }
        fclose(fptr) ;
        
        count-- ;
        input_data = (double **)malloc(sizeof(double *) * count) ;
        output_data = (double **)malloc(sizeof(double *) * count) ;
        num_data = 900 ;
        num_inputs = 10 ;

        fptr = fopen("cmc.txt" ,"r") ;
        if(fptr != NULL)
        {
            for(int i = 0 ; i < count ; i++)
            {
                input_data[i] = (double *)malloc(sizeof(double) * 9) ;
                output_data[i] = (double *)malloc(sizeof(double) * 3) ;
                double temp,red ;
                fscanf(fptr, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", &(input_data[i][0]), &(input_data[i][1]), &(input_data[i][2]), &(input_data[i][3]), &(input_data[i][4]), &(input_data[i][5]), &(input_data[i][6]), &(input_data[i][7]), &(input_data[i][8]), &temp) ;
                for(int j = 0 ; j < 3 ; j++)
                {
                    output_data[i][j] = 0.0 ;
                }
                if(temp == 1)
                {
                    output_data[i][0] = 1 ;
                }    
                else if(temp == 2)
                {
                    output_data[i][1] = 1 ;
                }    
                else
                {
                    output_data[i][2] = 1 ;
                }
                
                    
            }    
        }
        fclose(fptr) ;

        double *max_arr = (double *)malloc(sizeof(double) * 9) ;
        for(int i = 0 ; i < 9 ; i++)
            max_arr[i] = 0.0 ;
        for(int i = 0 ; i < count ; i++)
        {
            for(int j = 0 ; j < 9 ; j++ )
            {
                if(max_arr[j] < input_data[i][j])
                    max_arr[j] = input_data[i][j] ;
            }
        }
        for(int i = 0 ; i < count ; i++)
        {
            for(int j = 0 ; j < 9 ; j++ )
            {
                input_data[i][j] = input_data[i][j] / (double)max_arr[j] ;
            }
        }

    }

    Network *mlp = (Network *)malloc(sizeof(Network)) ;
    printf("\nDo not include the input layer in the number of layers .") ;
    printf("\nEnter the number of output classes as number of neurons in last layer .") ;
    printf("\nEnter the number of layers in the network : ") ;
    scanf("%d",&num_layers) ;

    Layer *arr_layer ;
    arr_layer = (Layer *)malloc(sizeof(Layer) * num_layers) ;
    mlp->num_layers = num_layers ;
    mlp->num_inputs = num_inputs ;
    mlp->arr_layer = arr_layer ;

    int prev_neuron_count = num_inputs ;
    for(int i = 1 ; i <= num_layers ; i++)
    {
        int neurons_layer ;
        printf("Enter the number of neurons in Layer %d :" ,i+1 ) ;
        scanf("%d",&neurons_layer) ;
        
        arr_layer[i-1].num_neurons = neurons_layer ;
        arr_layer[i-1].arr_neuron = (Neuron *)malloc(sizeof(Neuron) * neurons_layer) ;

        for(int j = 0 ; j < neurons_layer ; j++)
        {
            Neuron *curr_neuron = arr_layer[i-1].arr_neuron + j ;
            curr_neuron->size = prev_neuron_count + 1 ;
            curr_neuron->weights = (double *)malloc(sizeof(double) * curr_neuron->size) ;
            curr_neuron->prev_weights = (double *)malloc(sizeof(double) * curr_neuron->size) ;

            for(int k = 0 ; k < curr_neuron->size ; k++)
            {
                curr_neuron->weights[k] = (double)rand()/(double) RAND_MAX ;
                curr_neuron->prev_weights[k] = 0.0 ;
            }
        }
        prev_neuron_count = neurons_layer ;
    }

    int epochs,grad ;
    printf("\nEnter the number of epochs to be run : ") ;
    scanf("%d", &epochs) ;
    
    int k = 0 ;
    double err ;
    printf("\nEnter your choice for gradient descent : \n\t1: Regular Gradient Descent \n\t2: Gradient Decent with Momentum\n") ;
    scanf("%d",&grad) ;

    int ch ;
    printf("\nEnter the choice of activation function : \n\t1: Sigmoid \n\t2: Tanh \n") ;
    scanf("%d", &ch) ;
    act_choice = ch ;

    do{
        k++ ;
        err = avg_quad_err(mlp,input_data,output_data,num_data) ;
        printf("\nError : %lf " , err) ;

        for(int i = 0 ; i < num_data ; i++)
        {
            for(int j = mlp->num_layers-1 ; j >= 0 ; j--)
            {
                double *prev_layer_act = j>0 ? layer_output(mlp,input_data[i],j,1) : NULL ;
                double *delta = delta_calc(mlp,j+1,input_data[i],output_data[i]) ;
                if(grad == 1)
                    grad_desc_regular(mlp,j,input_data[i],output_data[i],delta,prev_layer_act) ;
                else if(grad == 2)
                    grad_desc_momentum(mlp,j,input_data[i],output_data[i],delta,prev_layer_act) ;    
            }
        }

        err = fabs(err - avg_quad_err(mlp,input_data,output_data,num_data)) ;
        // print_weights(mlp) ;

    }while(k < epochs && err > Epsilon) ;
    printf("\nEpochs : %d",k) ;
    print_acc(mlp,input_data,output_data,count,num_data+1) ;
    // print_weights(mlp) ;
}

double avg_quad_err(Network *mlp,double **input_data,double **output_data,int num_data) 
{

    double err = 0.0 ;
    int output_len = mlp->arr_layer[mlp->num_layers-1].num_neurons ;
    
    for(int i = 0 ; i < num_data ; i++)
    {
        double *calc_out = layer_output(mlp,input_data[i],mlp->num_layers,1) ;
        double e = 0.0 ;
        
        for(int j = 0 ; j < output_len ; j++ )
        {
            e += pow(output_data[i][j] - calc_out[j],2) ;
        }

        err += 0.5*e ;
    }    

    return err/num_data ;

}

double * layer_output(Network *mlp,double *input_data_single,int layer,int act)
{

    Layer *curr_layer = mlp->arr_layer + (layer-1) ;
	int last ;
    
    if(layer == mlp->num_layers)
        last = 0 ;
    else
        last = 1 ;

    if(act)
    {        
        int layer_size = curr_layer->num_neurons + last ;
        double *layer_wo_act = layer_output(mlp,input_data_single,layer,0) ;
        double *layer_act = (double *) malloc(sizeof(double) * layer_size) ;

        for(int i = 0 ; i < layer_size ;i++)
            if(!last)
                layer_act[i] = activation(layer_wo_act[i]) ;
            else
                layer_act[i] = i==0 ? -1.0 : activation(layer_wo_act[i-1]) ;
        return layer_act ;
    }
    else
    {
        int layer_size = curr_layer->num_neurons , neuron_size = curr_layer->arr_neuron->size ;
        double *layer_wo_act = (double *) malloc(sizeof(double)* layer_size ) ;
        double *prev_layer = layer <= 1 ? input_data_single : layer_output(mlp,input_data_single,layer-1,1) ;

        for(int i=0 ; i < layer_size ; i++)
        {
            layer_wo_act[i] = 0.0 ;
            Neuron *curr_neuron = curr_layer->arr_neuron + i ;
            for(int j = 0 ; j < neuron_size ; j++)
            {
                    layer_wo_act[i] += curr_neuron->weights[j] * prev_layer[j] ;
            }
        }
        return layer_wo_act ;
    }
    
}

double * delta_calc(Network *mlp, int layer, double *input_data_single, double *output_data_single)
{
	
    int layer_size = (mlp->arr_layer + layer - 1)->num_neurons ; 
	double *delta = (double *) malloc(sizeof(double) * layer_size);
	double *layer_wo_act = layer_output(mlp, input_data_single, layer , 0) ;

	if(mlp->num_layers == layer)
    {
		double *layer_act = layer_output(mlp, input_data_single, layer , 1) ;

		for(int i = 0 ; i < layer_size ; i++)
			delta[i] = (output_data_single[i] - layer_act[i]) * deriv_act(layer_wo_act[i]);
		
	} 
    else 
    {
		double *next_delta = delta_calc(mlp, layer+1, input_data_single, output_data_single) ;
		Layer *next_layer = mlp->arr_layer + layer ;

		for(int i = 0 ; i < layer_size ; i++){
			delta[i]=0.0 ;

			for(int j = 0 ; j < next_layer->num_neurons ; j++)
				delta[i] += next_delta[j] * *((next_layer->arr_neuron + j)->weights + i + 1) ;

			delta[i] *= deriv_act(layer_wo_act[i]) ;
		}

	}

	return delta;

}

void grad_desc_momentum(Network *mlp, int layer, double *input_data_single, double *output_data_single, double *delta, double *prev_layer_act)
{
	
    Layer *curr_layer = mlp->arr_layer + layer ;
	int layer_size = curr_layer->num_neurons ;

	for(int i = 0 ; i < layer_size ; i++)
    {
		Neuron *curr_neuron = curr_layer->arr_neuron + i ;

		for(int j = 0 ; j < curr_neuron->size ; j++)
        {
			
            // Momentum Term
			double mom_term = Momentum * (*(curr_neuron->weights + j) - *(curr_neuron->prev_weights + j)) ;
            if( *(curr_neuron->prev_weights + j) == 0)
                mom_term = 0 ;
			*(curr_neuron->prev_weights + j) = *(curr_neuron->weights + j) ;
            double grad_desc_step = (1 - Momentum) * (Rate * delta[i] * (layer==0 ? input_data_single[j] : prev_layer_act[j])) ;
			//Gradient Descent with Momentum Term
			*(curr_neuron->weights + j) = *(curr_neuron->weights + j) + mom_term + grad_desc_step ;

		}

	}

}

void grad_desc_regular(Network *mlp, int layer, double *input_data_single, double *output_data_single, double *delta, double *prev_layer_act)
{
	
    Layer *curr_layer = mlp->arr_layer + layer ;
	int layer_size = curr_layer->num_neurons ;

	for(int i = 0 ; i < layer_size ; i++)
    {
		Neuron *curr_neuron = curr_layer->arr_neuron + i ;

		for(int j = 0 ; j < curr_neuron->size ; j++)
        {
			//Regular Gradient Descent
			*(curr_neuron->weights + j) = *(curr_neuron->weights + j) + (Rate * delta[i] * (layer==0 ? input_data_single[j] : prev_layer_act[j])) ;
		}

	}
    
}

double activation(double x){
	
    if(act_choice == 1)
    {
        //Logistic
        return 1.0/(double)(1.0+exp(-x)) ;
    }
    else if(act_choice == 2)
    {
        //Hyperbolic Tangent
        return tanh(x);
    }
    else
    {
        return 0.0 ;
    }
    
}

double deriv_act(double x){
	
    if(act_choice == 1)
    {
        //Derivative of Logistic
        return activation(x)*(1.0-activation(x)) ;
    }
    else if(act_choice == 2)
    {
        // Derivative of Tanh
        double sech = 1.0 / cosh(x) ;
        return sech*sech ;
        return 1.0-pow(activation(x),2);
    }
    else
    {
        return 0.0 ;
    }
    
}

void act_softmax(double *output, int num_out)
{
    double out_max = -100000 ;

    for(int i = 0 ; i < num_out ; i++)
    {
        if(output[i] > out_max)
            out_max = output[i] ;
    }
    for(int i = 0 ; i < num_out ; i++)
    {
        if(output[i] == out_max)
            output[i] = 1 ;
        else
            output[i] = 0 ;
    }

}

void print_acc(Network *mlp, double **input_data, double **output_data, int num_data, int index) 
{
    int cor_data = 0 ;

    for(int i = index-1 ; i < num_data ; i++)
    {
        int num_out = (mlp->arr_layer + (mlp->num_layers - 1))->num_neurons ;
        double *out_wo_act = layer_output(mlp,input_data[i],mlp->num_layers ,1) ;
        act_softmax(out_wo_act,num_out) ;
        for(int j = 0 ; j < num_out ; j++)
        {
            if(output_data[i][j] == 1 && out_wo_act[j] == 1)
                cor_data++ ;
        }
    }

    printf("\nThe Test Accuracy of the network is %lf percent.",((double)cor_data/(num_data-index+1))*100 ) ;
    
}

void print_weights(Network *mlp)
{
    int num_layers = mlp->num_layers ;
    for(int i = 0 ; i< num_layers ; i++)
    {
        Layer *curr_layer = mlp->arr_layer + i ;
        int num_neurons = curr_layer->num_neurons ;
        printf("\nLayer %d : ",i+1) ;
        for(int j = 0 ; j < num_neurons ; j++ )
        {
            Neuron *curr_neuron = curr_layer->arr_neuron + j ;
            int num_weights = curr_neuron->size ;
            printf("\n\tNeuron %d : ", j+1 ) ;
            for(int k = 0 ; k < num_weights ; k++)
            {
                printf("\n\t\t %lf" ,*(curr_neuron->weights + k) ) ;
            }
        }
    }
}