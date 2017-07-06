#include<stdio.h>
#include <stdlib.h>
#include <math.h>
 #include <time.h>



// Max-min normalization of input data with m samples and d features
void min_max_normalize(double *xs, int m, int d)
{
        for (int x = 0; x < d; ++x) {
                // calculate std for each column
                double min = xs[x*d + 0];
                double max = xs[x*d + 0];
                for (int y = d; y < m*d; ++y) {
                        double val = xs[x*d + y];
                        if (val < min) {
                                min = val;
                        } else if (val > max) {
                                max = val;
                        }
                }

                for (int y = 0; y < m*d; ++y) {
                        double val = xs[x*d + y];
                        xs[x*d + y] = (val - min) / (max-min);
                }
        }
}


// CPU function for calculating the hypothesis function and individual gradient update for each feature of each sample
void map(int m, double *xs, double *ys, double *params, double *gradvec, int d){   // m is the no. of samples and d is the number of features in xs(input data)
        //double *h;
        //cudaMalloc (&h, m*sizeof(float));
        //int index = blockIdx.x * blockDim.x + threadIdx.x;

        for (int index=0; index<m; index++){
                double accum = params[0];
                //double accum = 0.0;
                for (int j=0; j<d; j++){
                        accum += xs[index*(d-1)+j] * params[j+1];
                }

                double h = 1.0/ (1.0 + exp(-accum));

                gradvec[index*d+0] =  (h - ys[index]) * 1;
		
                for (int j = 1; j < d; j++){
                        // gradvec[index*d+j] =  (h - ys[index]) * xs[index*d+j];
                        gradvec[j] +=  (h - ys[index]) * xs[index*d+j];
                }
        }
}



/*// Finds the final gradient by summing up the element-wise gradients columnwise

void reducegrad (double *gradvec, double *sumgradvec, int m, int d){
        
	for (int i=0; i<d; i++){
        	sumgradvec[i] = 0.0;
	}

        
        for (int index=0; index<m; index++){
		for (int index2=0; index2<m*d; index2+=d){
                	sumgradvec[index] += gradvec[index + index2];
		}
        }

        

}
*/


//Updates the weights/parameters based on the gradients
//alpha is the learning rate and lambda is the regularization parameter
void updateweight (double *params, double *sumgradvec, int m, int d, float alpha, float lambda){
         // For bias variable
	 params[0] = params[0] - alpha * (sumgradvec[0]) - lambda * alpha * params[0];

        for (int i=1; i<d; i++){
                params[i] = params[i] - alpha * (sumgradvec[i]) - lambda * alpha * params[i];
        }
}




#define num_iterations 2

int main(){
	clock_t start, end;
	double time_used;

        //Initialize number of samples and features
        int m = 1000000;
        int d = 401;

        //Allocate host memory variables
        size_t size1 = m*d*sizeof(double);
        size_t size2 = m*sizeof(double);
        size_t size3 = d*sizeof(double);

        double *xs;
        double *ys;
        double *params;
        double *sumgradvechost;
        double *gradvec;
        xs = (double*)malloc(size1);
        ys = (double*)malloc(size2);
        params = (double*)malloc(size3);
        sumgradvechost = (double*)malloc(size3);
        gradvec = (double*)malloc(size1);


        //Read input data from file
        FILE *fp, *fp1;
        fp = fopen ("input", "r");

        if (!fp){
                printf ("Unable to open file!");
        return 1;
        }

        for (int i=0; i<m; i++){
                for (int j=0; j<d-1; j++){
                        fscanf(fp, "%lf", &xs[i*(d-1) + j]);
                }
                fscanf(fp, "%lf", &ys[i]);
        }


        fclose(fp);

        //Initialize weights
        for (int i=0; i<d; i++){
                params[i] = 0.0;
	}

	 // Print first 10 rows of input data
        for (int i=0; i<20; i+=2) {
                printf("%lf %lf => %lf \n", xs[i], xs[i+1], ys[i/2]);
        }

      /*  //Max-min mormalize input data
        min_max_normalize(xs, m, d);
	*/

        
        

        for (int i=0; i<num_iterations; i++){
		start = clock();
                //Compute hypothesis function and element-wise gradients
                map(m, xs, ys, params, sumgradvechost, d);

                
                //Compute sum of all grad vector in GPU
                //reducegrad(gradvec, sumgradvechost, m, d);


                
		//Update weights in CPU. The learning rate is 0.001 and regualrization parameter is 10.
                updateweight(params, sumgradvechost, m, d, 0.001, 10);

                //Print current learned weights
                for (int j=0; j<d; j++){
                        printf("%lf \t", params[j]); }
                printf("\n");
		end = clock();
		time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
		printf("%f \n", time_used);

                
        }


        //Compute the predictions on the training data from the developed model
        double predict[m];
        for (int index=0; index<m; index++){
                predict[index] = params[0];
                for (int j=0; j<d; j++){
                        predict[index]  += xs[index*(d-1)+j] * params[j+1];
                }
        }



        //Compute the error for the model based on the percentage of true positives
        double error = 0.0;
        for (int i=0; i<m; i++){
                int tmp = 0;
                if ((1/( 1 + exp(-predict[i]))) >= 0.5) tmp = 1; else tmp = 0;
                if (tmp != ys[i])
                        error ++;
        }

        error = error / m;
        printf("%lf \n", error);


        //Dump the prediction output to a file
        fp1 = fopen("output", "w");
        for (int i=0; i<m; i++){
                fprintf(fp1, "%lf  \n", 1 / (1 + exp(-predict[i])));
        }


}


                                                                                                  




