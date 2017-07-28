#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>






__global__ void map1(int m, double *xs, double *weightih, double *hidden, int d, int n_hidden){   // m is the no. of samples and d is the number of features in xs(input data)
        
        int index = blockIdx.x * blockDim.x + threadIdx.x;
	//int j = blockIdx.x * blockDim.x + threadIdx.x;
	//int k = blockIdx.x * blockDim.x + threadIdx.x;


	if (index<m){
	
                
		for (int j = 0; j < n_hidden; j++){
			double accum = weightih[j];
                	for (int i=0; i<d; i++){
                        	accum += xs[index*d + i] * weightih[i*d + j];
                	}
			hidden[index*d + j] = 1.0/ (1.0 + exp(-accum));
		}
		

		/*for (int j=0; j<n_hidden; j++){
			double accum = matmul(xs, weightih, result, index, d);
			hidden[index*d + j] = 1.0/ (1.0 + exp(-accum);
		}
		*/
	}
}



__global__ void map2(int m, double *xs, double *ys, double *hidden, double *weightho, double *output, double *deltao, int d, int n_hidden, int n_output){
	//double error = 0.0;
	
        int index = blockIdx.x * blockDim.x + threadIdx.x;
	//int j = blockIdx.x * blockDim.x + threadIdx.x;
	//int k = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (index<m){
		for (int k = 0; k < n_output; k++){
			double accum = weightho[k];
                	for (int j=0; j<n_hidden; j++){
                        	accum += hidden[index*d + j] * weightho[j*d + k];
                	}
			output[index*d + k] = 1.0/ (1.0 + exp(-accum));
			//error[0] += (ys[index*d + k] - output[index*d + k]);
			deltao[k] = (ys[index*d + k] - output[index*d + k]) * output[index*d + k] * (1 - output[index*d + k]); 
		}
	}

	
}




__global__ void map3(int m, double *xs, double *ys, double *hidden, double *weightho, double *deltao, double *deltah, int d, int n_hidden, int n_output){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	//int j = blockIdx.x * blockDim.x + threadIdx.x;
	//int k = blockIdx.x * blockDim.x + threadIdx.x;



	if (index<m){
	for (int j = 0; j < n_hidden; j++){
	double accum = 0.0;
	for (int k = 0; k < n_output; k++){
		accum += weightho[j * d + k] * deltao[k];
	}
	deltah[j] = accum * hidden[index*d + j] * (1 - hidden[index*d + j]);
	}
	}	
}



__global__ void map4(int m, double *xs, double *ys, double eta, double *deltah, double *deltaweightih, double *weightih, int d, int n_hidden, int n_output){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	//int j = blockIdx.x * blockDim.x + threadIdx.x;
	//int i = blockIdx.x * blockDim.x + threadIdx.x;


	if (index<m){
	for (int j = 0; j < n_hidden; j++){
		deltaweightih[j] = eta * deltah[j];
	for (int i = 0; i < d; i++){
		deltaweightih[i * d + j] += eta * xs[index * d + i] * deltah[j];
		weightih[i * d + j] += deltaweightih[i * d + j];
	}
	
}
}
}

__global__ void map5(int m, double *xs, double *ys, double eta, double *hidden, double *deltao, double *deltah, double *deltaweightho, double *weightho, int d, int n_hidden, int n_output){
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index<m){
	for (int k = 0; k < n_output; k++){
		deltaweightho[k] = eta * deltao[k];
	for (int j = 0; j < n_hidden; j++){
		deltaweightho[j * d + k] += eta * hidden[index * d + j] * deltao[k];
		weightho[j * d + k] += deltaweightho[j * d + k];
	}
	
	}
	}
}



#define num_iterations 50
#define eta 0.5    // eta denotes the learning rate.
# include<time.h>
int main(){
        clock_t start, end;
        double time_used;

        //Initialize number of samples and features
        int n_patterns =  2500;
	int n_inputs = 20;
	int n_hidden = 10;
	int n_outputs = 1;


        //Allocate host memory variables
        size_t size1 = n_patterns * n_inputs * sizeof(double);
        size_t size2 = n_patterns * n_hidden * sizeof(double);
        size_t size3 = n_patterns * sizeof(double);
	size_t size4 = n_inputs * sizeof(double);
	size_t size5 = n_patterns * n_hidden * sizeof(double);
	size_t size6 = n_patterns * n_outputs * sizeof(double);	
	size_t size7 = n_inputs * n_hidden * sizeof(double);
	size_t size8 = n_hidden * n_outputs * sizeof(double);
	size_t size9 = n_outputs * sizeof(double);
	size_t size10 = n_hidden * sizeof(double);



        double *input;
	double *hidden;
	double *weightih;
	double *deltaweightih;
        double *weightho;
        double *deltaweightho;
        double *output;
	double *target;
	double *deltao;
	double *deltah;


        input = (double*)malloc(size1);
	hidden = (double*)malloc(size5);
        weightih = (double*)malloc(size7);
        deltaweightih = (double*)malloc(size7);
        weightho = (double*)malloc(size8);
        deltaweightho = (double*)malloc(size8);
	output = (double*)malloc(size6);
	target = (double*)malloc(size6);
	deltao = (double*)malloc(size9);
	deltah = (double*)malloc(size10);


        //Read input data from file
        FILE *fp, *fp1;
        fp = fopen ("input", "r");

        if (!fp){
                printf ("Unable to open file!");
        return 1;
        }

        for (int i=0; i<n_patterns; i++){
                for (int j=0; j<n_inputs; j++){
                        fscanf(fp, "%lf", &input[i*(n_inputs) + j]);
                }
                fscanf(fp, "%lf", &target[i]);
        }


        fclose(fp);

	

	for(int j = 0 ; j < n_hidden ; j++ ) {         /* initialize WeightIH and DeltaWeightIH */

		for(int i = 0 ; i < n_inputs ; i++ ) {
        		deltaweightih[i * n_inputs + j]= 0.0 ;
        		weightih[i * n_inputs + j] = 2.0 * ( rand()%n_patterns - 0.5 ) * 0.02 ;
    		}
	}


	/*for(int k = 0 ; k < n_outputs ; k ++ ) {         // initialize WeightHO and DeltaWeightHO 

		for(int j = 0 ; j < n_hidden ; j++ ) {
        		deltaweightho[j * n_hidden + k] = 0.0 ;
        		weightho[j * n_hidden + k] = 2.0 * ( rand()%n_patterns - 0.5 ) * 0.01 ;
    		}
	}
	*/

weightho[0] = 25.510000;
weightho[1] = 48.070000;
weightho[2] = 38.850000;
weightho[3] = 15.250000;
weightho[4] = 42.250000;
weightho[5] = 40.750000;
weightho[6] = 22.110000;
weightho[7] = 36.790000;
weightho[8] = 8.070000;
weightho[9] = 46.35000;


deltaweightho[0] = 0;
deltaweightho[1] = 0;
deltaweightho[2] = 0;
deltaweightho[3] = 0;
deltaweightho[4] = 0;
deltaweightho[5] = 0;
deltaweightho[6] = 0;
deltaweightho[7] = 0;
deltaweightho[8] = 0;
deltaweightho[9] = 0;


	for (int i=0; i<10; i++){
		printf("%lf \n", weightho[i]);
	}



	/*for (int i=0; i<n_patterns; i++){
        for (int j=0; j<n_hidden; j++){
                hidden[i*(n_hidden) + j] = 1.0;
        }
	}
	*/

	/*//Initialize weights
        for (int i=0; i<d; i++){
                params[i] = 0.0;
        }


	//Initialize nodes in each layer in the neural network
	float *out_input = (float *)malloc(sizeof(float) * (n_inputs + 1));
	float *out_hidden = (float *)malloc(sizeof(float) * n_hidden);
	float *out_output = (float *)malloc(sizeof(float) * n_outputs);

	buildLayer(out_input, n_inputs + 1, 1.0f);
	buildLayer(out_hidden, n_hidden, 1.0f);
	buildLayer(out_output, n_outputs, 1.0f);


	// Initialize changes layer
    	float *changes_input_hidden = buildWeightsLayer(n_inputs + 1, n_hidden, 0.0f);
    	float *changes_hidden_output = buildWeightsLayer(n_hidden, n_outputs, 0.0f);

    	// Initialize weight matrix
    	float *w_input_hidden = buildWeightsLayer(n_inputs + 1, n_hidden, -1.0f);
    	float *w_hidden_output = buildWeightsLayer(n_hidden, n_outputs, -1.0f);


        // Print first 10 rows of input data
        for (int i=0; i<20; i+=2) {
                printf("%lf %lf => %lf \n", xs[i], xs[i+1], ys[i/2]);
        }

	*/


	//Allocate variables in device memory
	double *input_d;
	double *hidden_d;
	double *weightih_d;
	double *deltaweightih_d;
	double *weightho_d;
	double *deltaweightho_d;
	double *output_d;
	double *target_d;
	double *deltao_d;
	double *deltah_d;
	double *error;


	
	cudaMalloc (&input_d , size1);
	cudaMalloc (&hidden_d , size5);
	cudaMalloc (&weightih_d , size7);
	cudaMalloc (&deltaweightih_d , size7);
	cudaMalloc (&weightho_d , size8);
	cudaMalloc (&deltaweightho_d , size8);
	cudaMalloc (&output_d , size6);
	cudaMalloc (&target_d , size6);
	cudaMalloc (&deltao_d , size9);
	cudaMalloc (&deltah_d , size10);
	cudaMalloc (&error, sizeof(double));



	
	//Copy vectors from host memory to device memory
	cudaMemcpy(input_d, input, size1, cudaMemcpyHostToDevice);
	//cudaMemcpy(output_d, output, size5, cudaMemcpyHostToDevice);
	//cudaMemcpy(hidden_d, hidden, size5, cudaMemcpyHostToDevice);
	cudaMemcpy(weightih_d, weightih, size7, cudaMemcpyHostToDevice);
	cudaMemcpy(deltaweightih_d, deltaweightih, size7, cudaMemcpyHostToDevice);
	cudaMemcpy(weightho_d, weightho, size8, cudaMemcpyHostToDevice);
	cudaMemcpy(deltaweightho_d, deltaweightho, size8, cudaMemcpyHostToDevice);
	//cudaMemcpy(output_d, deltaweightho, size8, cudaMemcpyHostToDevice);
	cudaMemcpy(target_d, target, size6, cudaMemcpyHostToDevice);
	//cudaMemcpy(deltao_d, deltao, size8, cudaMemcpyHostToDevice);
	//cudaMemcpy(deltah_d, deltah, size8, cudaMemcpyHostToDevice);


	//clock_t start, end;
	//double time_used;
	start = clock();

	for (int i=0; i<num_iterations; i++){
		cudaMemset((void*)error, 0, sizeof(double));
		printf("HI1");
		map1<<<2000,512>>>(n_patterns, input_d, weightih_d, hidden_d, n_inputs, n_hidden);
		printf("HI2");
		map2<<<2000,512>>>(n_patterns, input_d, target_d, hidden_d, weightho_d, output_d, deltao_d, n_inputs, n_hidden, n_outputs);
		//cudaMemcpy (output, output_d, size6, cudaMemcpyDeviceToHost);
		/*for (int j=0; j<10; j++){
			printf("%lf \n", weightho[j]);
		}*/
		
		printf("HI3");
		map3<<<2000,512>>>(n_patterns, input_d, target_d, hidden_d, weightho_d, deltao_d, deltah_d, n_inputs, n_hidden, n_outputs);
		printf("HI4");
		map4<<<2000,512>>>(n_patterns, input_d, target_d, eta, deltah_d, deltaweightih_d, weightih_d, n_inputs, n_hidden, n_outputs);
		printf("HI5");
		map5<<<2000,512>>>(n_patterns, input_d, target_d, eta, hidden_d, deltao_d, deltah_d, deltaweightho_d, weightho_d, n_inputs, n_hidden, n_outputs);
		printf("HI6");
		cudaMemcpy (weightih, weightih_d, size7, cudaMemcpyDeviceToHost);
		printf("HI7");
		cudaMemcpy (weightho, weightho_d, size8, cudaMemcpyDeviceToHost);
		printf("HI8");

	}

	end = clock();
	time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Time taken for copy in : %f \n", time_used);
	
	cudaMemcpy (output, output_d, size6, cudaMemcpyDeviceToHost);

	for (int i=0; i<10; i++){
		printf("%lf \n", weightih[i]);
	}

	for (int i=0; i<10; i++){
        printf("%lf \n", weightho[i]);
	}


	fp1 = fopen("nnet.out","w");	
	for (int i=0; i<2500;i++){
		fprintf(fp1, "%lf \n", output[i]);
	}
}	
