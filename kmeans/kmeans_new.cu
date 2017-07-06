#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <assert.h>
#define square(x) x*x
#include <float.h>

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



 

__global__ void getClusterCentroids(int n, double *xs, double *c, int k, int *s0, int *cluster_index, int d){ //xs indicates datapoints, c indicates centroids, k indicates no. of clusters
        
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index<n){
                double dist;
                double prevBest = DBL_MAX;
                int centroidIndex = 0;
                
                for (int clust = 0; clust < k; clust++) {
			dist = 0.0;
                	for (int dim=0; dim<d; dim++){
                        	dist += ((xs[index*d + dim]) - (c[clust*d+dim])) * ((xs[index*d + dim]) - (c[clust*d+dim]));
				
			}
			      
			if (dist<prevBest) {
			      prevBest = dist; centroidIndex = clust; 
			}
			
                }

		//atomicAddDouble(&s0[centroidIndex], 1.0);
		cluster_index[index] = centroidIndex;
	}
}




__global__ void calculateIntermediates(int n, double *xs, double *c, int k, int *cluster_index, int *intermediates0, double *intermediates1, double *intermediates2, int d){
     int blocksize = n / 450 + 1;
     int start = blockIdx.x * blocksize;
     int end1 = start + blocksize;
	int end;
	if (end1>n) end = n;
	else end = end1;

		if (end > n ) return;
		// loop for every K 
		for (int clust = threadIdx.y; clust < k; clust+= blockDim.y){
			// loop for every dimension(features)
			for (int dim = threadIdx.x; dim < d; dim+= blockDim.x) {

				// Calculate intermediate S0
				// for counts we don't have dimensions
				if (dim ==0) {
					int count = 0;
					for(int z=start; z<end; z++)
					{
						if(cluster_index[z] == clust) {
						count ++;
						}
					}
					intermediates0[blockIdx.x*k+clust] = count;
				}

				// Calculate intermediate S1 and S2
				double sum1 = 0.0;
				double sum2 = 0.0;
                                int idx ;
				for (int z=start; z<end; z++) {
					if(cluster_index[z] == clust) {
                                                idx = z * d + dim;
						sum1 += xs[idx];
						sum2 += xs[idx] * xs[idx];

					}
				}
				int index = (blockIdx.x*k*d + clust*d + dim);
                           	intermediates1[index] = sum1;
				intermediates2[index] = sum2;
			}
	}
}



__global__ void calculateFinal(int n, double *xs, double *c, int k, int *cluster_index, int *intermediates0, double *intermediates1, double *intermediates2, int *s0, double *s1, double *s2, int d){		// Only block is invoked.		

		// loop for every K 
	for (int clust = threadIdx.y; clust < k; clust+= blockDim.y){
		// loop for every dimension(features)
		for (int dim = threadIdx.x; dim < d; dim+= blockDim.x) {

			// Calculate  S0
			// for counts we don't have dimensions
			if (dim == 0) {
				//count = 0;
				for(int z = clust; z < 450*k; z+=k){
					{
						s0[clust] += intermediates0[z];
					}
				}
			}	

			// Calculate S1 and S2
            		int start = clust * d + dim;
            		int kd    = k * d;
            		double *s1end = &intermediates1[450 * kd];
            		double *s1cur = &intermediates1[start];
            		double *s2cur = &intermediates2[start];

	           	for (; s1cur < s1end; s1cur += kd, s2cur += kd)
	           	{
				s1[start] += *s1cur;
				s2[start] += *s2cur;
	           	}
 		

		}
	}
}		



void calculate_centroids (double *c1, int *s0, double *s1, double *s2, int k, int d, double cost){
	
	//cost = 0.0;
	for (int i = 0; i < k; i++){
		for (int j = 0; j < d; j++){
		if (s0[i] >= 1){
			c1 [i*d + j] = s1[i*d + j]/ s0[i];
		}
		else{
			c1 [i*d + j] = s1[i*d + j]/ 1;
		}
		
	}
	}
	
}


void calculate_cost (int n, double *xs, double *c1, int *s0, double *s1, double *s2, int k, int d, double cost){

        cost = 0.0;
       
        for (int i=0; i<k*d; i++){
        int mean = i/d;
        int x = s0[mean];
        double center;
        if (x>1){
                center = s1[i] / x;
        }
        else{
                center = s1[i];
        }

        cost += center * (center * x - 2 * s1[i]) + s2[i];
        }

        printf("COST: %lf \n", cost);
}
 



#include <time.h>

int main(int argc, char *argv[]){
        clock_t start, end;
        double time_used;

        int n, d, k, num_iterations;

        if(argc!=5){
                n = 200;
                d = 2;
                k = 2;
                num_iterations = 5;
        }

        else{
                n = atoi(argv[1]);
                d = atoi(argv[2]);
                k = atoi(argv[3]);
                num_iterations = atoi(argv[4]);
        }
	
	 //Allocate host memory variables
 	size_t size1 = n*d*sizeof(double);
 	size_t size2 = n*sizeof(double);
	size_t size3 = d*sizeof(double);
	size_t size4 = k*sizeof(int);
	size_t size5 = k*d*sizeof(double);
	size_t size6 = n*sizeof(int);
	size_t size7 = k*sizeof(int);
	size_t size8 = k*450*sizeof(int);
	size_t size9 = k*d*450*sizeof(double);



	double *xs;
	double *ys;
	int *cluster_index_host;
	int *s0_host;
	double *s1_host;
	double *s2_host;
	double *c_host;
	double *c1_host;
	double cost;

	double *gpu_xs;
	double *gpu_ys;
	int *cluster_index;
	double *c;
	int *s0;
 	double *s1;
	double *s2;
	int *intermediates0;
        double *intermediates1;
        double *intermediates2;
	double *intermediates1_host;
	


 	xs = (double*)malloc(size1);
 	ys = (double*)malloc(size2);
 	cluster_index_host = (int*)malloc(size6);
	c_host = (double*)malloc(size5);
	c1_host = (double*)malloc(size5);
	s0_host = (int*)malloc(size4);
	s1_host = (double*)malloc(size5);
	s2_host = (double*)malloc(size5);
	intermediates1_host = (double*)malloc(size9);
	
	cudaMalloc(&gpu_xs, size1);
	cudaMalloc(&gpu_ys, size2);
	cudaMalloc(&cluster_index, size6);
	cudaMalloc(&c, size5);
	cudaMalloc(&s0, size4);
	cudaMalloc(&s1, size5);
	cudaMalloc(&s2, size5);
	cudaMalloc(&intermediates0, size8);
	cudaMalloc(&intermediates1, size9);
	cudaMalloc(&intermediates2, size9);


	for (int i=0; i<k; i++){
		s0_host[i] = 0;
	}

	for (int i=0; i<k*d; i++){
		s1_host[i] = 0;
		s2_host[i] = 0;
	}



	//Read input data from file
	FILE *fp;
	fp = fopen ("kmeans_data", "r");

	if (!fp){
        	printf ("Unable to open file!");
		return 1;
	}

	for (int i=0; i<n; i++){
        	for (int j=0; j<d; j++){
                	fscanf(fp, "%lf", &xs[i*d + j]);
        	}	
        //fscanf(fp, "%lf", &ys[i]);
	}


	fclose(fp);

		

	//Randomly select k datapoints as centroids

	int ind[2];

	for (int i=0; i<k; i++){
		ind[i] = rand()%n;
	}
	
	
	
	for (int i=0; i<k; i++){
		for (int j=0; j<d; j++){
			int r = ind[i];
			c_host[i*d + j] = xs[r*d + j];
		}
	}

		
	
	start = clock();
	cudaMemcpy(c, c_host, size5, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_xs, xs, size1, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_ys, ys, size2, cudaMemcpyHostToDevice);
	cudaMemcpy(s0, s0_host, size4, cudaMemcpyHostToDevice);
	cudaMemcpy(s1, s1_host, size5, cudaMemcpyHostToDevice);
	cudaMemcpy(s2, s2_host, size5, cudaMemcpyHostToDevice);
	end = clock();
	time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Time taken for copy in : %f \n", time_used);



	
	int changed = 1;
	
	while (changed==1){
	for (int i=0; i<num_iterations; i++){
          	start = clock();
		cudaMemset((void*)s0, 0, size4);
                //Compute hypothesis function and element-wise gradients
                getClusterCentroids<<<2,100>>>(n, gpu_xs, c, k, s0, cluster_index, d);


                //Copy the element wise gradients from GPU to CPU
                //cudaMemcpy(gradvec1, gradvec, size1, cudaMemcpyDeviceToHost);

                //Compute sum of all grad vector in GPU
		cudaMemset((void*)s1, 0, size5);
		cudaMemset((void*)s2, 0, size5);
              	cudaMemset((void*)intermediates0, 0, size8);
		cudaMemset((void*)intermediates1, 0, size9);
		cudaMemset((void*)intermediates2, 0, size9);
	
		dim3 nthreads(d,k);
		calculateIntermediates<<<450,nthreads>>>(n, gpu_xs, c, k, cluster_index, intermediates0, intermediates1, intermediates2, d);		

		cudaMemcpy(intermediates1_host, intermediates1, size5, cudaMemcpyDeviceToHost);

		
		dim3 nthreads1(d,k);
		calculateFinal<<<1,nthreads1>>>(n, gpu_xs, c, k, cluster_index, intermediates0, intermediates1, intermediates2, s0, s1, s2, d);
		
		cudaMemcpy(s0_host, s0, size4, cudaMemcpyDeviceToHost);
	
		
	
		cudaMemcpy(s1_host, s1, size5, cudaMemcpyDeviceToHost);
 		
		cudaMemcpy(s2_host, s2, size5, cudaMemcpyDeviceToHost);

			
		calculate_centroids (c1_host, s0_host, s1_host, s2_host, k, d, cost);

		calculate_cost (n, xs, c1_host, s0_host, s1_host, s2_host, k, d, cost);
		
		
		double maxdelta = 0.0;

		for (int i=0; i<k; i++){
			for (int j=0; j<d; j++){
				maxdelta += (c1_host[i*d+j] - c_host[i*d+j]) * (c1_host[i*d+j] - c_host[i*d+j]);
			}
		}


	
		memcpy(c_host, c1_host, size5);

		changed = maxdelta>0.5;	
		cudaMemcpy(c, c1_host, size5, cudaMemcpyHostToDevice);
		

		end = clock();
		time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
		printf("Time taken for map1 : %f \n", time_used);



				
	}
	}
	

	cudaMemcpy(cluster_index_host, cluster_index, size6, cudaMemcpyDeviceToHost);


	for (int i=0; i<5; i++){
		printf("%d \n", cluster_index_host[i]);
	}

	for (int i=140; i<150; i++){
        printf("%d \n", cluster_index_host[i]);
	}

	for (int i=0; i<k; i++){
		printf("%d \n", s0_host[i]);
	}


	FILE *fp1;
	//Dump the prediction output to a file
        fp1 = fopen("output_kmeans", "w");
        for (int i=0; i<n; i++){
                fprintf(fp1, "%d  \n", cluster_index_host[i]);
        }
}

			


	























