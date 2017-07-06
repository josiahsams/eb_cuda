1. The file data_create.c contains the C code for generating the input dataset used.

2. The file parallel_logistic_regression.cu contains the CUDA C implementation for parallelized logistic regression with regularizer to be run on GPUs.
  
   In order to run this program, we need to type the following command in the  CUDA terminal (compute capability >3.0) with arg1 = number of samples, arg2 = number of features and arg3 = number fo iterations

	 nvcc -arch=sm_35 parallel_logistic_regression.cu -o parallel_logistic_regression.ouT

   In order to get the output, type the following command in the terminal:
	
	./parallel_logistic_regression.out

3. The file parallel_logistic_regression.c contains the corresponding C code for implementing logistic regression in a serailized fashion. This in turn will help us to compare the performance of the parallelized version of logistic regression a lgorithm with the serial one. 



