#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <time.h>

#define WARP_SIZE 32
#define DEBUG false






/* ---------------- [[HELPER FUNCTIONS FOR GLOBAL MEMORY]] ---------------- */

float *_copyHostDevice(float *src, int src_size) {
    float *src_d;
    cudaMalloc((void**)&src_d, sizeof(float) * src_size);
    cudaMemcpy(src_d, src, sizeof(float) * src_size, cudaMemcpyHostToDevice);
    return src_d;
}

float *_copyDeviceHost(float *src, int src_size, float *dst=NULL) {
    float *target;
    if (dst == NULL) {
        target = (float*)malloc(sizeof(float) * src_size);
    } else {
        target = dst;
    }
    
    cudaMemcpy(target, src, sizeof(float) * src_size, cudaMemcpyDeviceToHost);
    return target;
}

/* ---------------- [[HELPER FUNCTIONS FOR TILING]] ---------------- */

typedef struct {
    int x;
    int y;
} GlobalDim;



__device__ GlobalDim getGlobalDim(dim3 blockDim, dim3 blockIdx, dim3 threadIdx) {
    GlobalDim gd;
    gd.x = blockDim.x * blockIdx.x + threadIdx.x;
    gd.y = blockDim.y * blockIdx.y + threadIdx.y;
    return gd;
}

dim3 getGridBasedOnBlockSize(int width, int height, int block_size) {
    int gridX = (int)ceil((float)width / block_size);
    int gridY = (int)ceil((float)height / block_size);
    return dim3(gridX, gridY);
}

/* ---------------- [[HELPER FUNCTIONS FOR DEBUGGING]] ---------------- */

void _sleep(int n) {
    #ifdef __APPLE__
        sleep(n);
    #endif
}

void drawMatrix(float *m, int width, int height) {
    for (int i=0; i < height; i++) {
        for (int j=0; j < width; j++) {
            printf("%f ", m[i * width + j]);
        }
        printf("\n");
    }
}






/* ---------------- [[CUDA KERNELS]] ---------------- */

__global__ void updateWeightsCUDA(int n_inputs, int n_outputs, float *weights, float *changes, float *delta_outputs, float *inputs) {
    int width = n_outputs;
    int height = n_inputs;
    GlobalDim gd = getGlobalDim(blockDim, blockIdx, threadIdx);

    if ((gd.x < width) && (gd.y < height)) {
        int idx = width * gd.y + gd.x;
        float change = delta_outputs[gd.x] * inputs[gd.y];
        
        weights[idx] += 0.5 * change + 0.5 * changes[idx];
        changes[idx] = change;
    }

}

__global__ void mapStepCUDA(float *inputs, float *matrix, float *buffer, int width, int height) {
    GlobalDim gd = getGlobalDim(blockDim, blockIdx, threadIdx);

    if ((gd.x < width) && (gd.y < height)) {
        int idx = width * gd.y + gd.x;
        buffer[idx] = inputs[gd.y] * matrix[idx];
    }
}

__global__ void reduceStepCUDA(float *input, float *output, int width, int height) {

    __shared__ float sharedMemory[WARP_SIZE * WARP_SIZE];

    // STEP 1: exclude all threads that do not depend from problem
    GlobalDim gd = getGlobalDim(blockDim, blockIdx, threadIdx);


    if ((gd.x < width) && (gd.y < height)) {

        // STEP 2: Move to shared memory
        int gridId = gd.y * width + gd.x;
        int blockId = threadIdx.y * blockDim.x + threadIdx.x;
        sharedMemory[blockId] = input[gridId];
        __syncthreads();

        int n = (int)ceil((float)blockDim.y/2);
        while(n >= 1) {
            if (threadIdx.y < n) {

                if ((gd.y + n) < height) {
                    int firstIndex = blockId;
                    int secondIndex = blockDim.x * (threadIdx.y + n) + threadIdx.x;
                    sharedMemory[firstIndex] += sharedMemory[secondIndex];
                }
            }
            __syncthreads();
            if (n == 1) {
                break;
            } else {
                n = (int)ceil((float)n/2);
            }
        }
        __syncthreads();

        // STEP 3: Write back results
        if (threadIdx.y == 1) {
            output[blockIdx.y * width + gd.x] = sharedMemory[threadIdx.x];
        }
    }
}

/* ---------------- [[LAUNCH FUNCTIONS]] ---------------- */

void setWeightsForLayers(float *weights, float *changes, float *delta_outputs, float *inputs, int n_inputs, int n_outputs) {

    // Copy to device memory
    int grid_size = n_inputs * n_outputs;
    float *weights_d = _copyHostDevice(weights, grid_size);
    float *changes_d = _copyHostDevice(changes, grid_size);
    float *delta_outputs_d = _copyHostDevice(delta_outputs, n_outputs);
    float *inputs_d = _copyHostDevice(inputs, n_inputs);

    // Define block structure
    dim3 block(WARP_SIZE, WARP_SIZE);
    dim3 grid = getGridBasedOnBlockSize(n_outputs, n_inputs, WARP_SIZE);

    // RUN RUN RUN!
    updateWeightsCUDA<<<grid, block>>>(n_inputs, n_outputs, weights_d, changes_d, delta_outputs_d, inputs_d);

    // Copy back weights and momenutm
    weights = _copyDeviceHost(weights_d, grid_size, weights);
    changes = _copyDeviceHost(changes_d, grid_size, changes);
}


void update_layer(float *src_layer, float *dst_layer, int src_n, int dst_n, float *weights) {
    dim3 block(WARP_SIZE, WARP_SIZE);

    float *src_layer_d, *weights_d, *buffer_d;
    int total = src_n * dst_n;
 
    // Allocate input in global memory
    src_layer_d = _copyHostDevice(src_layer, src_n);
    weights_d = _copyHostDevice(weights, total);
    cudaMalloc((void**)&buffer_d, sizeof(float) * total);
 
    // Create block dimensions and run parallel update layer
    int gridX = (int)ceil((float)dst_n/WARP_SIZE);
    int gridY = (int)ceil((float)src_n/WARP_SIZE);
    dim3 grid(gridX, gridY);

    // RUN RUN RUN!
    if (DEBUG) {
        printf("\n***** Updating layer *****\n");

        printf("\nFrom\n");
        drawMatrix(src_layer, src_n, 1);

        printf("\nTo\n");
        drawMatrix(weights, dst_n, src_n);
    }
    mapStepCUDA<<<grid, block>>>(src_layer_d, weights_d, buffer_d, dst_n, src_n);

    // Set the current target to the input
    float *currentTarget = buffer_d;
    int currentHeight = src_n;

    while (currentHeight > 1) {

        // Calculate grid size
        int gridX = (int)ceil((float)dst_n/WARP_SIZE);
        int gridY = (int)ceil((float)currentHeight/WARP_SIZE);
        dim3 grid(gridX, gridY);

        // Allocate new buffer
        float *buffer_d;
        cudaMalloc((void**)&buffer_d, sizeof(float) * (dst_n * gridY));
 
        // RUN RUN RUN!
        reduceStepCUDA<<<grid, block>>>(currentTarget, buffer_d, dst_n, currentHeight);

        // Free old memory and keep track of the new one
        cudaFree(currentTarget);
        currentHeight = grid.y;
        currentTarget = buffer_d;
    }

    dst_layer =_copyDeviceHost(currentTarget, dst_n, dst_layer);
    for (int i=0; i < dst_n; i++) {
        dst_layer[i] = tanh(dst_layer[i]);
    }

    if (DEBUG) {
        printf("\nResult is\n");
        drawMatrix(dst_layer, dst_n, 1);
        printf("\n***** ENDED UPDATING LAYER *****\n");
        _sleep(1);
    }
}






#ifdef __APPLE__
    #include <unistd.h>
#endif

typedef struct {
    int n_inputs;
    int n_hidden;
    int n_outputs;
    
    float *out_input;
    float *out_hidden;
    float *out_output;

    float *changes_input_hidden;
    float *changes_hidden_output;
    
    float *w_input_hidden;
    float *w_hidden_output;
} NeuralNet;

typedef struct {
    int *result;
    int *data;
} Pattern;

void buildLayer(float *arr, int n, float initial) {
    int i=0;
    while(i < n){
        *arr = initial;
        arr++;
        i++;
    }
}

float* buildWeightsLayer(int outer_n, int inner_n, float seed) {
    int total = outer_n * inner_n;
    float *w = (float *)malloc(sizeof(float) * total);
    for(int i=0; i < total; i++) {
        if (seed == -1) {
          w[i] = ((float)rand()/(float)RAND_MAX);
        } else {
          w[i] = seed;
        }
    }
    return w;
}

NeuralNet buildNeuralNet(int n_inputs, int n_outputs, int n_hidden) {

    float *out_input = (float *)malloc(sizeof(float) * (n_inputs + 1));
    float *out_hidden = (float *)malloc(sizeof(float) * n_hidden);
    float *out_output = (float *)malloc(sizeof(float) * n_outputs);

    buildLayer(out_input, n_inputs + 1, 1.0f);
    buildLayer(out_hidden, n_hidden, 1.0f);
    buildLayer(out_output, n_outputs, 1.0f);
    
    // Build changes layer
    float *changes_input_hidden = buildWeightsLayer(n_inputs + 1, n_hidden, 0.0f);
    float *changes_hidden_output = buildWeightsLayer(n_hidden, n_outputs, 0.0f);
    
    // Build weight matrix
    float *w_input_hidden = buildWeightsLayer(n_inputs + 1, n_hidden, -1.0f);
    float *w_hidden_output = buildWeightsLayer(n_hidden, n_outputs, -1.0f);

    NeuralNet nn;

    nn.n_inputs = n_inputs + 1;
    nn.n_outputs = n_outputs;
    nn.n_hidden = n_hidden;

    nn.out_input = out_input;
    nn.out_hidden = out_hidden;
    nn.out_output = out_output;

    nn.changes_input_hidden = changes_input_hidden;
    nn.changes_hidden_output = changes_hidden_output;

    nn.w_input_hidden = w_input_hidden;
    nn.w_hidden_output = w_hidden_output;

    return nn;
}

float dsigmoid(float y) {
    return 1 - pow(y,2.0f);
}

void update_pattern(Pattern pattern, NeuralNet nn) {

    if (DEBUG) {
        printf("\n ***** LAYER UPDATE *****\n");
    }

    // Write inputs
    int i;
    for(i=0; i < nn.n_inputs -1; i++) {
        nn.out_input[i] = pattern.data[i];
    }

    // Run parallel update
    update_layer(nn.out_input, nn.out_hidden, nn.n_inputs, nn.n_hidden, nn.w_input_hidden);
    update_layer(nn.out_hidden, nn.out_output, nn.n_hidden, nn.n_outputs, nn.w_hidden_output);

    if (DEBUG) {
        printf("\n ***** END LAYER UPDATE *****\n");
    }
}

float back_propagate_network(Pattern p, NeuralNet n) {

    if (DEBUG) {
        printf("\n ***** BACK PROPAGATE *****\n");
    }

    int i, j;
    float *output_delta = (float*)malloc(sizeof(float) * n.n_outputs);
    float *hidden_delta = (float*)malloc(sizeof(float) * n.n_hidden);

    
    // Calculate output delta
    for (i=0; i < n.n_outputs; i++) {
        float error = p.result[i] - n.out_output[i];
        output_delta[i] = dsigmoid(n.out_output[i]) * error;
    }
    
    
    // Calculate hidden delta
    for(i=0; i < n.n_hidden; i++) {
        float error = 0.0f;
        for (j=0; j < n.n_outputs; j++) {
            error += output_delta[j] * n.w_hidden_output[i * n.n_outputs + j];
        }
        hidden_delta[i] = dsigmoid(n.out_hidden[i]) * error;
    }

    // Set hidden-output weights
    setWeightsForLayers(n.w_hidden_output, n.changes_hidden_output, output_delta, n.out_hidden, n.n_hidden, n.n_outputs);
    if (DEBUG) {
        printf("\nHidden-Output weights\n");
        drawMatrix(n.w_hidden_output, n.n_outputs, n.n_hidden);
        _sleep(1);
    }
   
    setWeightsForLayers(n.w_input_hidden, n.changes_input_hidden, hidden_delta, n.out_input, n.n_inputs, n.n_hidden);
    if (DEBUG) {
        printf("\nInput-Hidden weights\n");
        drawMatrix(n.w_input_hidden, n.n_hidden, n.n_inputs);
        _sleep(1);
    }
   
    // Calculate error
    float error = 0.0f;
    for (i=0; i < n.n_outputs; i++) {
        error = error + 0.5f * pow(p.result[i] - n.out_output[i], 2);
    }
    if (DEBUG) {
        printf("\n ***** Error for this pattern is: %f *****\n", error); 
        _sleep(2);
    }
    return error;
}


void train_network(Pattern *patterns, int n_patterns, int n_iterations, NeuralNet nn) {
  int i, j;
  for (i=0; i < n_iterations; i++) {
    float error = 0;
    for (j=0; j < n_patterns; j++) {
       update_pattern(patterns[j], nn);
       error += back_propagate_network(patterns[j], nn);
    }
    if (i % 10 == 0) {
       printf("Error is: %-.5f\n", error);
       if (DEBUG) _sleep(2);
    }
  }
}

Pattern makePatternSingleOutput(int *data, int sample_no, int dim, int result) {
    Pattern p;
    //p.data = data;
    //for (int i=0; i<n_samples; i++){
    int data1[dim];
    for (int j=0; j<dim; j++){
		data1[j] = data[sample_no*dim + j];
    }
    p.data = data1;

    p.result = (int *)malloc(sizeof(int));
    p.result[0] = result;

    return p;
}

#include <time.h>

int main() {
    clock_t start, end;
    double time_used;

    srand((unsigned)time(NULL));
    
    int n_inputs = 20;
    int n_hidden = 10;
    int n_outputs = 1;
    
    // Build output layer
    NeuralNet nn = buildNeuralNet(n_inputs, n_outputs, n_hidden);
    
    // Build training samples
    /*int _p1[] = {0,0};
    Pattern p1 = makePatternSingleOutput(_p1, 0);
    int _p2[] = {0,1};
    Pattern p2 = makePatternSingleOutput(_p2, 1);
    int _p3[] = {1,1};
    Pattern p3 = makePatternSingleOutput(_p3, 0);
    int _p4[] = {1,0};
    Pattern p4 = makePatternSingleOutput(_p4, 1);
*/
	Pattern p[2500];
	
	int _p[50000];
	
	for (int i=0; i<2500; i++){
		for (int j=0; j<20; j++){
			_p[i * 20 + j] = i+j;
		}
	}

	
	for (int i=0; i<2500; i++){
				
		if (i%2==0)
			p[i] = makePatternSingleOutput(_p, i, 20, 0);
		else
			p[i] = makePatternSingleOutput(_p, i, 20, 1);
	}

    
   // Pattern patterns[] = {p3, p2, p1, p4};

	//Pattern patterns[] = {p[2], p[1], p[0], p[3]};
    
    // Train the network
start = clock();
    train_network(p, 2500, 1, nn);
end = clock();
	time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
printf("Time taken for training the neural net : %f \n", time_used);

    printf("\n\nTesting the network\n");
    //update_pattern(p2, nn);
	update_pattern(p[1], nn);
    for (int i=0; i < nn.n_outputs; i++) {
        //printf("Output: %f, expected: %i\n", nn.out_output[i], p2.result[i]);
	printf("Output: %f, expected: %i\n", nn.out_output[i], p[1].result[i]);
    }
    cudaDeviceReset();
    return 0;

}
