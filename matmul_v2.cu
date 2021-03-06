#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_
 
#include <stdio.h>
 
// Thread block size
#define BLOCK_SIZE 16
#define TILE_SIZE  16
 
#define WA 1024   // Matrix A width
#define HA 1024   // Matrix A height
#define WB 1024   // Matrix B width
#define HB WA     // Matrix B height
#define WC WB     // Matrix C width
#define HC HA     // Matrix C height
 
// CUDA Kernel
__global__ void matrixMul( float* C, float* A, float* B, int wA, int wB)
{
 
   // 1. 2D Thread ID
   int tx = blockIdx.x * TILE_SIZE + threadIdx.x;
   int ty = blockIdx.y * TILE_SIZE + threadIdx.y;
 
   // value stores the element that is 
   // computed by the thread
   float value = 0;
   for (int i = 0; i < wA; ++i)
   {
      float elementA = A[ty * wA + i];
      float elementB = B[i * wB + tx];
      value += elementA * elementB;
   }
 
   // Write the matrix to device memory each 
   // thread writes one element
   C[ty * wA + tx] = value;
}
 
#endif // #ifndef _MATRIXMUL_KERNEL_H_








// Multiply two matrices A * B = C
 
#include <stdlib.h>

#include <math.h>

 
// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
   for (int i = 0; i < size; ++i)
   data[i] = rand() / (float)RAND_MAX;
}
 
/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////
 
int
main(int argc, char** argv)
{

   // set seed for rand()
   srand(2006);
 
   // 1. allocate host memory for matrices A and B
   unsigned int size_A = WA * HA;
   unsigned int mem_size_A = sizeof(float) * size_A;
   float* h_A = (float*) malloc(mem_size_A);
 
   unsigned int size_B = WB * HB;
   unsigned int mem_size_B = sizeof(float) * size_B;
   float* h_B = (float*) malloc(mem_size_B);
 
   // 2. initialize host memory
   randomInit(h_A, size_A);
   randomInit(h_B, size_B);

/*	for (int i=0; i<WA; i++){
		for (int j=0; j<HA; j++){
			if (i==j) h_A[i] = 1;
			else h_A[i] = 0;
		}
	}



	for (int i=0; i<WB; i++){
        for (int j=0; j<HB; j++){
                if (i==j) h_B[i] = 1;
                else h_B[i] = 0;
        }
	}
*/
 
   // 3. print out A and B
/*   printf("\n\nMatrix A\n");
   for(int i = 0; i < size_A; i++)
   {
      printf("%f ", h_A[i]);
      if(((i + 1) % WA) == 0)
      printf("\n");
   }
  
   printf("\n\nMatrix B\n");
   for(int i = 0; i < size_B; i++)
   {
      printf("%f ", h_B[i]);
      if(((i + 1) % WB) == 0)
      printf("\n");
   }
 */
   // 8. allocate device memory
   float* d_A;
   float* d_B;
   cudaMalloc((void**) &d_A, mem_size_A);
   cudaMalloc((void**) &d_B, mem_size_B);
 
   // 9. copy host memory to device
   cudaMemcpy(d_A, h_A, mem_size_A, 
   cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, h_B, mem_size_B, 
   cudaMemcpyHostToDevice);
 
   // 4. allocate host memory for the result C
   unsigned int size_C = WC * HC;
   unsigned int mem_size_C = sizeof(float) * size_C;
   float* h_C = (float*) malloc(mem_size_C);
 
   // 10. allocate device memory for the result
   float* d_C;
   cudaMalloc((void**) &d_C, mem_size_C);
 
   // 5. perform the calculation
   // setup execution parameters
   dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
   dim3 grid(WC / threads.x, HC / threads.y);
 
   // execute the kernel
   matrixMul<<< grid, threads >>>(d_C, d_A, 
                                  d_B, WA, WB);
 
   // 11. copy result from device to host
   cudaMemcpy(h_C, d_C, mem_size_C, 
   cudaMemcpyDeviceToHost);
 
   // 6. print out the results
   printf("\n\nMatrix C (Results)\n");
   for(int i = 0; i < size_C; i++)
   {
      printf("%f ", h_C[i]);
      if(((i + 1) % WC) == 0)
      printf("\n");
   }
   printf("\n");
 


	FILE *fp;
	fp = fopen("test","w");
	for (int i=0; i<1024*1024; i++){
		fprintf(fp,"%f \n", h_C[i]);
	}

   // 7. clean up memory
   free(h_A);
   free(h_B);
   free(h_C);
   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);
 
}
