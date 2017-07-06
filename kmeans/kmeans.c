#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>



void map(int n, double *xs, double *c, int k, double *s0, int *cluster_index, int d){ //xs indicates datapoints, c indicates centroids, k indicates no. of clusters

        //int index = blockIdx.x * blockDim.x + threadIdx.x;

        for (int index = 0; index < n; index++){
                double dist;
                double prevBest = 10000000000.0;
                int centroidIndex = 0;

                for (int j1 = 0; j1 < k; j1++) {
                        dist = 0.0;
                        for (int j=0; j<d; j++){
                                dist += ((xs[index*d + j]) - (c[j1*d+j])) * ((xs[index*d + j]) - (c[j1*d+j]));
                                //if(index == 0 ) printf("%lf \t %lf \t", xs[index*d + j], c[j1*d+j]);
                        }
                              //if(index == 0 ) printf("%d: %d -  %d :: %f == %f \n", j1, centroidIndex, j1, dist, prevBest);
                        if (dist<prevBest) {
                              prevBest = dist; centroidIndex = j1;
                        }

                }

                //atomicAddDouble(&s0[centroidIndex], 1.0);
		s0[centroidIndex]++;	
                cluster_index[index] = centroidIndex;
        }
}




void map1(int n, double *xs, double *c, int k, int *cluster_index, double *s1, double *s2, int d){ //xs indicates datapoints, c indicates centroids, k indicates no. of cluster

        //int index = blockIdx.x * blockDim.x + threadIdx.x;

        for (int index = 0; index < n; index++){
                int ind = cluster_index[index];

                for (int j = 0; j < d; j++){
                                s1[ind*d + j] += xs[index*d + j];
                                s2[ind*d + j] += xs[index*d + j] * xs[index*d + j];
                                //atomicAddDouble(&s1[ind*d + j] , xs[index*d + j]);
                                //atomicAddDouble(&s2[ind*d + j] , xs[index*d + j] * xs[index*d + j]);
                                //if(index == 0 ) printf("%d \t %lf \t %lf \n ", cluster_index[index], xs[index*d + j], s1[ind*d + j], xs[index*d+j]);
                       }
                         //if(index == 0 ) printf("%d \t %lf \t %lf \n ", cluster_index[index], s1[ind*d + 0], xs[index*d+0]);
                }
}


void calculate_centroids (double *c1, double *s0, double *s1, double *s2, int k, int d, double cost){

        //cost = 0.0;
        for (int i = 0; i < k; i++){
                for (int j = 0; j < d; j++){
                if (s0[i] >= 1){
                        c1 [i*d + j] = s1[i*d + j]/ s0[i];
                }
                else{
                        c1 [i*d + j] = s1[i*d + j]/ 1;
                }
                //cost += (c1[i*d + j] * ((c1[i*d + j] * s0[i]) - 2*s1[i])) + s2[i];
        }
        }
        //printf("%lf \n", cost);
}



void calculate_cost (int n, double *xs, double *c1, double *s0, double *s1, double *s2, int k, int d, double cost){

        cost = 0.0;
        /*for (int i = 0; i < n; i++){
                for (int j = 0; j < d; j++){
                int cluster = cluster_index[i];
                cost[0] += (xs[i*d+j] - c1[cluster*d+j]) * (xs[i*d+j] - c1[cluster*d+j]);
        }
        }*/


        for (int i=0; i<k*d; i++){
        int mean = i/d;
        double x = s0[mean];
        double center;
        if (x>1){
                center = s1[i] / x;
        }
        else{
                center = s1[i];
        }

        cost += center * (center * x - 2 * s1[i]) + s2[i];
        }

        printf("COSSTT: %lf \n", cost);
}




#define num_iterations 5
#include <time.h>

int main(){
        clock_t start, end;
        double time_used;

        int n = 200;
        int d = 2;
        int k = 2;

         //Allocate host memory variables
        size_t size1 = n*d*sizeof(double);
        size_t size2 = n*sizeof(double);
        size_t size3 = d*sizeof(double);
        size_t size4 = k*sizeof(double);
        size_t size5 = k*d*sizeof(double);
        size_t size6 = n*sizeof(int);
        size_t size7 = k*sizeof(int);


        double *xs;
        double *ys;
        int *cluster_index_host;
        double *s0_host;
        double *s1_host;
        double *s2_host;
        double *c_host;
        double *c1_host;
        double cost;

	

	xs = (double*)malloc(size1);
	ys = (double*)malloc(size2);
	cluster_index_host = (int*)malloc(size6);
	c_host = (double*)malloc(size5);
	c1_host = (double*)malloc(size5);
	s0_host = (double*)malloc(size4);
	s1_host = (double*)malloc(size5);
	s2_host = (double*)malloc(size5);


	for (int i=0; i<k; i++){
        	s0_host[i] = 0;
	}

	for (int i=0; i<k*d; i++){
        	s1_host[i] = 0;
        	s2_host[i] = 0;
	}

	
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

	printf("HI1");
 	for (int i=0; i<5; i++){
         	for (int j=0; j<d; j++){
                	printf("%lf \n", xs[i*d + j]);
        	}
	}


	//Randomly select k datapoints as centroids

	int ind[2];

	/*for (int i=0; i<k; i++){
        	ind[i] = rand()%n;
         	//printf ("%d \t", ind[i]);
 	}
	*/

	ind[0] = 5; ind[1] = 70;



	for (int i=0; i<k; i++){
        	for (int j=0; j<d; j++){
                	int r = ind[i];
                	c_host[i*d + j] = xs[r*d + j];
        	}
	}

	

	for (int i=0; i<k; i++){
		for (int j=0; j<d; j++){
        		printf("%lf \n", c_host[i*d + j]);
		}
	}


	printf("HI");

	
//double cost = 10000000.0;
//double cost1 = 0.0;
//int changed = 1;

	//while (changed==1){
	for (int i=0; i<num_iterations; i++){
        	start = clock();
        	memset((void*)s0_host, 0, size4);
        	//Compute hypothesis function and element-wise gradients
        	map(n, xs, c_host, k, s0_host, cluster_index_host, d);
        	/*end = clock();
        	time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        	printf("%f \n", time_used);
*/

        	//Copy the element wise gradients from GPU to CPU
        	//cudaMemcpy(gradvec1, gradvec, size1, cudaMemcpyDeviceToHost);

        	//Compute sum of all grad vector in GPU
        	//start = clock();
        	memset((void*)s1_host, 0, size5);
        	memset((void*)s2_host, 0, size5);
        	map1(n, xs, c_host, k, cluster_index_host, s1_host, s2_host, d);
        	/*end = clock();
        	time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        	printf("%f \n", time_used);
		*/

        	//Calculate centroids
                //start = clock();
                calculate_centroids (c1_host, s0_host, s1_host, s2_host, k, d, cost);
		calculate_cost (n, xs, c1_host, s0_host, s1_host, s2_host, k, d, cost);
		/*end = clock();
		time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
		printf("%f \n", time_used);
		*/

                /*for (int i=0; i<k; i++){
                        for (int j=0; j<d; j++){
                                printf("%lf \n", c1_host[i*d + j]);
                        }
                }*/

                //start = clock();
                double maxdelta = 0.0;

                for (int i=0; i<k; i++){
                        for (int j=0; j<d; j++){
                                maxdelta += (c1_host[i*d+j] - c_host[i*d+j]) * (c1_host[i*d+j] - c_host[i*d+j]);
                        }
                }
                end = clock();
 		time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
		printf("%f \n", time_used);

		memcpy(c_host, c1_host, size5);


	}
		
	
		for (int i=0; i<5; i++){
                printf("%d \n", cluster_index_host[i]);
        }

        for (int i=140; i<150; i++){
        printf("%d \n", cluster_index_host[i]);
        }

        for (int i=0; i<k; i++){
                printf("%lf \n", s0_host[i]);
        }


        FILE *fp1;
        //Dump the prediction output to a file
        fp1 = fopen("output_kmeans", "w");
        for (int i=0; i<n; i++){
                fprintf(fp1, "%d  \n", cluster_index_host[i]);
        }

	


	
}	

