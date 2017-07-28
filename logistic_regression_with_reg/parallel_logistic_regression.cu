#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <assert.h>



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


// GPU function for calculating the hypothesis function and individual gradient update for each feature of each sample
__global__ void map(int m, double *xs, double *ys, double *params, double *gradvec, int d){   // m is the no. of samples and d is the number of features in xs(input data)
	//double *h;
	//cudaMalloc (&h, m*sizeof(float));
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index<m){
		double accum = params[0];
		//double accum = 0.0;
                for (int j=0; j<d; j++){
                        accum += xs[index*(d-1)+j] * params[j+1];
                }
		
		double h = 1.0/ (1.0 + exp(-accum));
	
		gradvec[index*d+0] =  (h - ys[index]) * 1;

		for (int j = 1; j < d; j++){
			gradvec[index*d+j] =  (h - ys[index]) * xs[index*d+j];
		}	
	}
}




#define WARPSIZE  32

__device__ inline double atomicAddDouble(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed  = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + 
                      __longlong_as_double(assumed)));
  } while (assumed != old);

  return __longlong_as_double(old);
}

__device__ inline double __shfl_double(double d, int lane) {
  // Split the double number into 2 32b registers.
  int lo, hi;
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(d));

  // Shuffle the two 32b registers.
  lo = __shfl(lo, lane);
  hi = __shfl(hi, lane);

  // Recreate the 64b number.
  asm volatile("mov.b64 %0, {%1,%2};" : "=d"(d) : "r"(lo), "r"(hi));
  return d;
}

__device__ inline double warpReduceSum(double val) {
  int i = blockIdx.x  * blockDim.x + threadIdx.x;
#pragma unroll
  for (int offset = WARPSIZE / 2; offset > 0; offset /= 2) {
     val += __shfl_double(val, (i + offset) % WARPSIZE);
  }
  return val;
}

__device__ inline double4 __shfl_double4(double4 d, int lane) {
  // Split the double number into 2 32b registers.
  int lox, loy, loz, low, hix, hiy, hiz, hiw;
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lox), "=r"(hix) : "d"(d.x));
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(loy), "=r"(hiy) : "d"(d.y));
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(loz), "=r"(hiz) : "d"(d.z));
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(low), "=r"(hiw) : "d"(d.w));

  // Shuffle the two 32b registers.
  lox = __shfl(lox, lane);
  hix = __shfl(hix, lane);
  loy = __shfl(loy, lane);
  hiy = __shfl(hiy, lane);
  loz = __shfl(loz, lane);
  hiz = __shfl(hiz, lane);
  low = __shfl(low, lane);
  hiw = __shfl(hiw, lane);

  // Recreate the 64b number.
  asm volatile("mov.b64 %0, {%1,%2};" : "=d"(d.x) : "r"(lox), "r"(hix));
  asm volatile("mov.b64 %0, {%1,%2};" : "=d"(d.y) : "r"(loy), "r"(hiy));
  asm volatile("mov.b64 %0, {%1,%2};" : "=d"(d.z) : "r"(loz), "r"(hiz));
  asm volatile("mov.b64 %0, {%1,%2};" : "=d"(d.w) : "r"(low), "r"(hiw));
  return d;
}

__device__ inline double4 warpReduceVSum(double4 val4) {
  int i = blockIdx.x  * blockDim.x + threadIdx.x;
#pragma unroll
  for (int offset = WARPSIZE / 2; offset > 0; offset /= 2) {
     double4 shiftedVal4 = __shfl_double4(val4, (i + offset) % WARPSIZE);
     val4.x += shiftedVal4.x;
     val4.y += shiftedVal4.y;
     val4.z += shiftedVal4.z;
     val4.w += shiftedVal4.w;
  }
  return val4;
}



__device__ double* deviceReduceKernelj(double * inArray, double *out, long i, long n, long length) {
    double sum = 0;
    double *inArrayBody;
    int index =  blockIdx.x * blockDim.x + threadIdx.x;
    for (long idx = index; idx < n; idx += blockDim.x * gridDim.x) {
        inArrayBody = &inArray[idx*length];
        sum += inArrayBody[i];
    }

    sum = warpReduceSum(sum);

    if ((threadIdx.x & (WARPSIZE -1)) == 0){
        atomicAddDouble(out, sum);
    }
    return out;
}




__device__ void deviceReduceArrayKernelj(double * inArray, double *outputArrayBody, long length, long n) {
    long i = 0;
    double *inArrayBody;

    // unrolled version
    while ((length - i) >= 4) {
        double4 sum4;
        sum4.x = 0; sum4.y = 0; sum4.z = 0; sum4.w = 0;
        for (long idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
            inArrayBody = &inArray[idx*length];
            sum4.x += inArrayBody[i];
            sum4.y += inArrayBody[i+1];
            sum4.z += inArrayBody[i+2];
            sum4.w += inArrayBody[i+3];
        }

        sum4 = warpReduceVSum(sum4);

        if ((threadIdx.x & (WARPSIZE - 1)) == 0) {

        double *outx = &outputArrayBody[i];
        double *outy = &outputArrayBody[i+1];
        double *outz = &outputArrayBody[i+2];
        double *outw = &outputArrayBody[i+3];
            atomicAddDouble(outx, sum4.x);
            atomicAddDouble(outy, sum4.y);
            atomicAddDouble(outz, sum4.z);
            atomicAddDouble(outw, sum4.w);
        }
        i += 4;
    }

    for (; i < length; i++) {
        deviceReduceKernelj(inArray, &outputArrayBody[i], i, n, length);
    }
}

// Finds the final gradient by summing up the element-wise gradients columnwise
extern "C"
__global__
void reducegrad(double *gradvec, double * sumgradvec, int m, int d) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < m)
       deviceReduceArrayKernelj(gradvec, sumgradvec, d, m);

}


//Updates the weights/parameters based on the gradients
//alpha is the learning rate and lambda is the regularization parameter
void updateweight (double *params, double *sumgradvec, int m, int d, float alpha, float lambda){
	for (int i=0; i<d; i++){
		params[i] = params[i] - alpha * (sumgradvec[i]) - lambda * alpha * params[i];
	}
}


#define num_iterations 2
# include<time.h>

int main(){
	clock_t start, end;
     	double time_used;

	//Initialize number of samples and features
	int m =  1000000;
	int d = 401;

	//Allocate host memory variables
	size_t size1 = m*d*sizeof(double);
 	size_t size2 = m*sizeof(double);
 	size_t size3 = d*sizeof(double);
	
	double *xs;
	double *ys;
	double *params;
	double *sumgradvechost;
	double *gradvec1;
	xs = (double*)malloc(size1);
        ys = (double*)malloc(size2);
	params = (double*)malloc(size3);
	sumgradvechost = (double*)malloc(size3);
	gradvec1 = (double*)malloc(size1);


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

/*	//Max-min mormalize input data
	min_max_normalize(xs, m, d);	

	//Print first 10 rows of input data after normalization
	for (int i=0; i<20; i+=2) {
      		printf("%lf %lf => %lf \n", xs[i], xs[i+1], ys[i/2]);
	}
*/
	
	//Allocate variables in device memory
	double *gpu_params;
	double *gpu_xs;
	double *gpu_ys;
	double *gradvec;
	double *sumgradvec;
	cudaMalloc (&gpu_params, size3);
	cudaMalloc(&gpu_xs, size1);
	cudaMalloc(&gpu_ys, size2);
	cudaMalloc(&gradvec, size1);
	cudaMalloc(&sumgradvec, size3);


	//Copy vectors from host memory to device memory
	cudaMemcpy(gpu_xs, xs, size1, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_ys, ys, size2, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_params, params, size3, cudaMemcpyHostToDevice);


	for (int i=0; i<num_iterations; i++){
		start = clock();	
		//Compute hypothesis function and element-wise gradients
		map<<<1000,1024>>>(m, gpu_xs, gpu_ys, gpu_params, gradvec, d);
		end = clock();
		time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("%f \n", time_used);
		//Copy the element wise gradients from GPU to CPU
		//cudaMemcpy(gradvec1, gradvec, size1, cudaMemcpyDeviceToHost);

 		//Compute sum of all grad vector in GPU
		start = clock();
		reducegrad<<<1000,1024>>>(gradvec, sumgradvec, m, d);
		end = clock();
		time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
		printf("%f \n", time_used);
		
		start = clock();
 		//Copy out grad's vector from GPU to CPU
 		cudaMemcpy (sumgradvechost, sumgradvec, sizeof(double)*d, cudaMemcpyDeviceToHost);
		 end = clock();
 time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
 printf("%f \n", time_used);


 		//Update weights in CPU. The learning rate is 0.001 and regualrization parameter is 10.
		start = clock();
		updateweight(params, sumgradvechost, m, d, 0.001, 10);
		 end = clock();
 time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
 printf("%f \n", time_used);


	 	//Print current learned weights
		for (int j=0; j<d; j++){
         		printf("%lf \t", params[j]); }
         	printf("\n");
		
         
 		// Copy in the updated weights back to GPU
		start = clock();
 		cudaMemcpy (gpu_params, params, sizeof(double) * d, cudaMemcpyHostToDevice);
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


