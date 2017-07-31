#include <cassert>
#include <cstddef>
#include <vector>
#include <memory>
#include <cmath>
#include <vector>
#include <iostream>
#include <cstdio>
#include <mutex>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>



// The default num threads per 2D block.
const int TPB_X = 32;
const int TPB_Y = 32;


#define CheckError(ans)                                                                            \
  { neuralnetwork::cuda::util::OutputError((ans), __FILE__, __LINE__); }




namespace math {

struct MatrixView {
  unsigned rows;
  unsigned cols;
  float *data; // row major order.

  static MatrixView Create(unsigned rows, unsigned cols) {
    assert(rows > 0 && cols > 0);

    MatrixView result;
    result.rows = rows;
    result.cols = cols;
    result.data = new float[rows * cols];
    return result;
  }

  static void Release(MatrixView &view) {
    assert(view.data != nullptr);
    delete[] view.data;
    view.data = nullptr;
  }
};
}




namespace neuralnetwork {
namespace cuda {

enum class LayerActivation { TANH, LOGISTIC, RELU, LEAKY_RELU, LINEAR, SOFTMAX };


struct LayerWeights {
	unsigned inputSize; // this includes the bias. So it should be equal to prev layer size + 1
	unsigned layerSize;

	// Data pointers allocated with cudaMallocPitch. Logical size is (inputSize * layerSize)
	// num rows = layerSize, num cols = inputSize
	float *weights;

  	// The pitch of the rows of the weights matrix in bytes.
  	size_t pitch;

  	__device__ float *Elem(unsigned r, unsigned c) {
		assert(r < layerSize && c < inputSize);
		return (float *)((char *)weights + r * pitch) + c;
  	}
};

struct Random {
  curandState *d_state;
  unsigned numStates;

  __device__ void Initialise(unsigned threadIndex, unsigned seed) {
    curand_init(seed, threadIndex, 0, &d_state[threadIndex]);
  }

  __device__ float SampleUniform(unsigned threadIndex) {
    return curand_uniform(&d_state[threadIndex % numStates]);
  }

  static Random Create(unsigned numStates, unsigned seed);
  static void Cleanup(Random &rnd);
};



struct SamplesBatch {
  unsigned maxBatchSize; // number of rows allocated in memory.
  unsigned batchSize;    // equal to the number of rows in the matrix.
  unsigned inputDim;     // equal to the number of columns in the matrix.
  unsigned targetOutputDim;

  float *input; // matrix sized batchSize(rows) * sampleDim(cols)
  size_t ipitch;

  float *targetOutput; // matrix sized batchSize(rows) * sampleDim(cols)
  size_t opitch;

  __device__ float *InputElem(unsigned r, unsigned c) {
    assert(r < maxBatchSize && c < inputDim);
    return (float *)((char *)input + r * ipitch) + c;
  }

  __device__ float *TargetOutputElem(unsigned r, unsigned c) {
    assert(r < maxBatchSize && c < targetOutputDim);
    return (float *)((char *)targetOutput + r * opitch) + c;
  }
};





struct LayerBatchOutputs {
  unsigned maxBatchSize;
  unsigned batchSize;

  // layer size includes the bias term, so it will be equal to the number of nodes + 1
  unsigned layerSize;

  float *output; // matrix sized batchSize(rows) * layerSize(cols)
  size_t opitch;

  float *derivative; // matrix sized batchSize(rows) * layerSize(cols)
  size_t dpitch;

  __device__ float *OutputElem(unsigned r, unsigned c) {
    assert(r < maxBatchSize && c < layerSize);
    return (float *)((char *)output + r * opitch) + c;
  }

  __device__ float *DerivativeElem(unsigned r, unsigned c) {
    assert(r < maxBatchSize && c < layerSize);
    return (float *)((char *)derivative + r * dpitch) + c;
  }
};



struct LayerBatchDeltas {
  unsigned maxBatchSize;
  unsigned batchSize;
  unsigned layerSize;

  float *delta; // matrix sized batchSize(rows) * layerSize(cols)
  size_t pitch;

  __device__ float *Elem(unsigned r, unsigned c) {
    assert(r < maxBatchSize && c < layerSize);
    return (float *)((char *)delta + r * pitch) + c;
  }
};


namespace GradientKernel {

void Apply(LayerBatchDeltas layerDeltas, LayerBatchOutputs layerOutputs, LayerWeights outGradient,
           cudaStream_t stream);
}



namespace ForwardPassKernel {

void Apply(LayerWeights layerWeights, LayerBatchOutputs input, LayerBatchOutputs output,
           LayerActivation activation, Random rnd, float nodeActivationRate, bool isOutputLayer,
           cudaStream_t stream);
}



namespace BackwardDeltaKernel {

void Apply(LayerBatchDeltas nextDelta, LayerWeights transposedWeights,
           LayerBatchOutputs layerOutput, LayerBatchDeltas outDelta, cudaStream_t stream);
}


struct NetworkSpec {
  unsigned numInputs;
  unsigned numOutputs;
  std::vector<unsigned> hiddenLayers;
  float nodeActivationRate;

  unsigned maxBatchSize;

  LayerActivation hiddenActivation;
  LayerActivation outputActivation;
};


class CudaNetwork {
public:
  CudaNetwork(const NetworkSpec &spec);
  ~CudaNetwork();

  void SetWeights(const std::vector<math::MatrixView> &weights);
  void GetWeights(std::vector<math::MatrixView> &outWeights);

  void Train(const math::MatrixView &batchInputs, const math::MatrixView &batchOutputs);

private:
  struct CudaNetworkImpl;
  std::unique_ptr<CudaNetworkImpl> impl;
};


namespace util {

void OutputError(cudaError_t code, const char *file, int line);

void *AllocPushBuffer(size_t bufSize);
void FreePushBuffer(void *buf);

LayerWeights NewLayerWeights(unsigned inputSize, unsigned layerSize);
void DeleteLayerWeights(LayerWeights &lw);

SamplesBatch NewSamplesBatch(unsigned maxBatchSize, unsigned inputDim, unsigned targetOutputDim);
void DeleteSamplesBatch(SamplesBatch &sb);

LayerBatchOutputs NewLayerBatchOutputs(unsigned maxBatchSize, unsigned layerSize);
void DeleteLayerBatchOutputs(LayerBatchOutputs &lbo);

LayerBatchDeltas NewLayerBatchDeltas(unsigned maxBatchSize, unsigned layerSize);
void DeleteLayerBatchDeltas(LayerBatchDeltas &lbd);

void PrintMatrixView(math::MatrixView view);
void PrintLayerWeights(LayerWeights d_weights);
void PrintLayerOutputs(LayerBatchOutputs d_outputs);
void PrintLayerDeltas(LayerBatchDeltas d_deltas);
}



namespace TransposeKernel {

void Apply(LayerWeights layerWeights, LayerWeights transposedWeights, cudaStream_t stream);
}


namespace SoftmaxKernel {

void Apply(const LayerBatchOutputs &lastLayer, cudaStream_t stream);
}

}
}




using namespace neuralnetwork;
using namespace neuralnetwork::cuda;

__device__ float activationValue(float in, const LayerActivation activation) {
  switch(activation) {
  case LayerActivation::TANH:
    return tanhf(in);
  case LayerActivation::LOGISTIC:
    return 1.0f / (1.0f + expf(-in));
  case LayerActivation::RELU:
    return fmaxf(0.0f, in);
  case LayerActivation::LEAKY_RELU:
    return fmaxf(0.01f * in, in);
  case LayerActivation::LINEAR:
  case LayerActivation::SOFTMAX:
    return in;
  }
  assert(false); // should never get here.
  return in;
}



__device__ float activationDerivative(float in, float out, const LayerActivation activation) {
	switch(activation) {
		case LayerActivation::TANH:
			return 1.0f - out * out;
  		case LayerActivation::LOGISTIC:
    			return out * (1.0f - out);
  		case LayerActivation::RELU:
    			return in > 0.0f ? 1.0f : 0.0f;
  		case LayerActivation::LEAKY_RELU:
    			return in > 0.0f ? 1.0f : 0.01f;
  		case LayerActivation::LINEAR:
  		case LayerActivation::SOFTMAX:
    	return 1.0f;
  	}
  	assert(false); // should never get here.
  	return 1.0f;
}



__global__ void transposeKernel(LayerWeights lw, LayerWeights out, unsigned bufStride) {
  extern __shared__ float buf[];

  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < lw.inputSize && y < lw.layerSize) {
    buf[threadIdx.x + threadIdx.y * bufStride] = *lw.Elem(y, x);
  }

  __syncthreads();

  x = blockIdx.y * blockDim.y + threadIdx.x;  // transpose block offset
  y = blockIdx.x * blockDim.x + threadIdx.y;

  if (x < out.inputSize && y < out.layerSize) {
    *(out.Elem(y, x)) = buf[threadIdx.y + threadIdx.x * bufStride];
  }
}




__global__ void forwardPassKernel(LayerWeights lw, LayerBatchOutputs prevOutputs,
                                  LayerBatchOutputs out, const LayerActivation activation,
                                  Random rnd, const float nodeActivationRate, const bool isOutput,
                                  const unsigned spitch) {

  extern __shared__ float buf[]; // shared memory buffer

  const unsigned row = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned col = blockDim.x * blockIdx.x + threadIdx.x;

  // TODO: can implement a "fast path" and "slow path" versions of the below code and branch here.
  // Fast path can assume that the entire block will fall within the bounds of all of the matrices
  // and dispense with a whole bunch of the below checks.

  const int numChunks = (lw.inputSize + blockDim.x - 1) / blockDim.x;

  // buffer for holding the layer weight matrix chunk
  float *lwChunk = (float *) buf;

  // buffer for holding the prev outputs matrix chunk
  float *poChunk = (float *) &buf[spitch * blockDim.y];

  float sum = 0.0f;
  const int lwRow = blockDim.x * blockIdx.x + threadIdx.y;
  const int poRow = row;

  const int chunkIndex = threadIdx.x + threadIdx.y * spitch;
  const int lim = numChunks * blockDim.x;

  for (int chunkOffset = 0; chunkOffset < lim; chunkOffset += blockDim.x) {
    const int lwCol = chunkOffset + threadIdx.x;
    if (lwRow < lw.layerSize && lwCol < lw.inputSize) {
      lwChunk[chunkIndex] = *lw.Elem(lwRow, lwCol);
    }

    const int poCol = lwCol;
    if (poRow < prevOutputs.batchSize && poCol < prevOutputs.layerSize) {
      poChunk[chunkIndex] = *prevOutputs.OutputElem(poRow, poCol);
    }
    __syncthreads();

    int chunkLim = min(blockDim.x, lw.inputSize - chunkOffset);
    for (int j = 0; j < chunkLim; j++) {
      sum += lwChunk[j + threadIdx.x * spitch] * poChunk[j + threadIdx.y * spitch];
    }
    __syncthreads();
  }

  if (row < out.batchSize && col < out.layerSize - 1) {
    float *outElem = out.OutputElem(row, col);
    float *dElem = out.DerivativeElem(row, col);

    if (isOutput || rnd.SampleUniform(col + row * out.layerSize) < nodeActivationRate) {
      *outElem = activationValue(sum, activation);
      *dElem = activationDerivative(sum, *outElem, activation);
    } else {
      *outElem = 0.0f;
      *dElem = 0.0f;
    }
  }
}





// computes outDelta = tw * nextDelta (elemwisemul) layerOutput.derivatives
__global__ void backwardDeltaKernel(LayerBatchDeltas nextDelta, LayerWeights tw,
                                    LayerBatchOutputs layerOutput, LayerBatchDeltas outDelta,
                                    unsigned spitch) {

  extern __shared__ float buf[]; // shared memory buffer

  const unsigned row = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned col = blockDim.x * blockIdx.x + threadIdx.x;

  const int numChunks = (tw.inputSize + blockDim.x - 1) / blockDim.x;

  // buffer for holding the layer weight matrix chunk
  float *twChunk = (float *) buf;

  // buffer for holding the prev outputs matrix chunk
  float *ndChunk = (float *) &buf[spitch * blockDim.y];

  const int twRow = blockDim.x * blockIdx.x + threadIdx.y;
  const int ndRow = row;

  const int chunkIndex = threadIdx.x + threadIdx.y * spitch;
  const int lim = numChunks * blockDim.x;

  float sum = 0.0f;
  for (int chunkOffset = 0; chunkOffset < lim; chunkOffset += blockDim.x) {
    const int twCol = chunkOffset + threadIdx.x;
    if (twRow < tw.layerSize && twCol < tw.inputSize) {
      twChunk[chunkIndex] = *tw.Elem(twRow, twCol);
    }

    const int ndCol = twCol;
    if (ndRow < nextDelta.batchSize && ndCol < nextDelta.layerSize) {
      ndChunk[chunkIndex] = *nextDelta.Elem(ndRow, ndCol);
    }
    __syncthreads();

    int chunkLim = min(blockDim.x, tw.inputSize - chunkOffset);
    for (int j = 0; j < chunkLim; j++) {
      sum += twChunk[j + threadIdx.x * spitch] * ndChunk[j + threadIdx.y * spitch];
    }
    __syncthreads();
  }

  if (row < outDelta.batchSize && col < outDelta.layerSize) {
    float od = *layerOutput.DerivativeElem(row, col);
    *outDelta.Elem(row, col) = sum * od;
  }
}







__global__ void gradientKernel(LayerBatchDeltas layerDeltas, LayerBatchOutputs layerOutputs,
                               LayerWeights outGradient, unsigned spitch) {
  extern __shared__ float buf[]; // shared memory buffer

  const unsigned row = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned col = blockDim.x * blockIdx.x + threadIdx.x;

  // buffer for holding the layer weight matrix chunk
  float *ldChunk = (float *) buf;

  // buffer for holding the prev outputs matrix chunk
  float *loChunk = (float *) &buf[spitch * blockDim.y];

  const int ldCol = blockDim.y * blockIdx.y + threadIdx.x;
  const int loCol = col;

  const int numChunks = (layerDeltas.batchSize + blockDim.y - 1) / blockDim.y;
  const int chunkIndex = threadIdx.x + threadIdx.y * spitch;
  const int lim = numChunks * blockDim.y;

  float sum = 0.0f;
  for (int chunkOffset = 0; chunkOffset < lim; chunkOffset += blockDim.y) {
    const int ldRow = chunkOffset + threadIdx.y;
    if (ldRow < layerDeltas.batchSize && ldCol < layerDeltas.layerSize) {
      ldChunk[chunkIndex] = *layerDeltas.Elem(ldRow, ldCol);
    }

    const int loRow = ldRow;
    if (loRow < layerOutputs.batchSize && loCol < layerOutputs.layerSize) {
      loChunk[chunkIndex] = *layerOutputs.OutputElem(loRow, loCol);
    }
    __syncthreads();

    int chunkLim = min(blockDim.x, layerDeltas.batchSize - chunkOffset);
    for (int j = 0; j < chunkLim; j++) {
      sum += ldChunk[threadIdx.y + j * spitch] * loChunk[threadIdx.x + j * spitch];
    }

    __syncthreads();
  }

  if (row < outGradient.layerSize && col < outGradient.inputSize) {
    *outGradient.Elem(row, col) = sum / layerDeltas.batchSize;
  }
}




// This softmax code assumes that the output layer is smaller than the maximum number of threads
// in a block. For ease of implementation, we assume this and do the whole thing in a single block.
// This allows easy synchronization and easy algorithm. Most problems wont have >1024 outputs.
// Separate blocks can do separate batch rows.
__global__ void softmaxKernel(LayerBatchOutputs outputs) {
  extern __shared__ float buf[]; // shared memory buffer

  const unsigned outIndex = threadIdx.x;
  const unsigned batchIndex = blockIdx.x;

  assert(blockDim.x <= outputs.layerSize && gridDim.x == outputs.batchSize);

  // A single float to hold data to exchange between threads in this block.
  float *sharedVar = (float *) &buf[0];

  // Buffer to hold all of the output elements for this batch element.
  float *outElems = (float *) &buf[1];

  // 1. Copy the row for the current batch into shared memory.
  float val = *(outputs.OutputElem(batchIndex, outIndex));
  outElems[outIndex] = val;
  __syncthreads();

  // 2. Find the max element in the row, done by a single thread per block while all others wait.
  float maxValue;
  if (outIndex == 0) {
    maxValue = outElems[0];
    for (unsigned i = 1; i < blockDim.x; i++) {
      maxValue = fmaxf(maxValue, outElems[i]);
    }
    *sharedVar = maxValue;
  }
  __syncthreads();
  maxValue = *sharedVar;

  // 3. Calc the unnormalised exponent offset by the max value and write it to shared mem.
  val = expf(val - maxValue);
  outElems[outIndex] = val;
  __syncthreads();

  // 4. Calculate the sum across the batch, done by a single thread per block.
  float sum = 0.0f;
  if (outIndex == 0) {
    for (unsigned i = 0; i < blockDim.x; i++) {
      sum += outElems[i];
    }
    *sharedVar = sum;
  }
  __syncthreads();
  sum = *sharedVar;

  // 5. Calc the normalised value for each output elem and write it out to global mem.
  *(outputs.OutputElem(batchIndex, outIndex)) = val / sum;
}








void TransposeKernel::Apply(LayerWeights layerWeights, LayerWeights transposedWeights,
                            cudaStream_t stream) {
  int bpgX = (layerWeights.inputSize + TPB_X - 1) / TPB_X;
  int bpgY = (layerWeights.layerSize + TPB_Y - 1) / TPB_Y;

  unsigned stride = TPB_X + 1;
  size_t sharedMemSize = stride * TPB_Y * sizeof(float);

  transposeKernel<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1), sharedMemSize, stream>>>(
      layerWeights, transposedWeights, stride);
}


void ForwardPassKernel::Apply(LayerWeights layerWeights, LayerBatchOutputs input,
                              LayerBatchOutputs output, LayerActivation activation,
                              Random rnd, float nodeActivationRate, bool isOutputLayer,
                              cudaStream_t stream) {
  assert(layerWeights.inputSize == input.layerSize);
  assert(layerWeights.layerSize == output.layerSize - 1);

  // -1 is here since we dont need to compute the bias term for the output vector.
  int bpgX = (output.layerSize - 1 + TPB_X - 1) / TPB_X;
  int bpgY = (output.batchSize + TPB_Y - 1) / TPB_Y;

  unsigned spitch = (TPB_X + 1);
  size_t sharedMemSize = 2 * spitch * TPB_Y * sizeof(float);

  forwardPassKernel<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1), sharedMemSize, stream>>>(
      layerWeights, input, output, activation, rnd, nodeActivationRate, isOutputLayer, spitch);
}




void BackwardDeltaKernel::Apply(LayerBatchDeltas nextDelta, LayerWeights transposedWeights,
                                LayerBatchOutputs layerOutput, LayerBatchDeltas outDelta,
                                cudaStream_t stream) {

  // TODO: handle bank conflicts. Do the same in the forward kernel.
  assert(nextDelta.layerSize == transposedWeights.inputSize);
  assert(outDelta.layerSize == transposedWeights.layerSize - 1);
  assert(outDelta.layerSize == layerOutput.layerSize - 1);
  assert(nextDelta.batchSize == layerOutput.batchSize);
  assert(nextDelta.batchSize == outDelta.batchSize);

  int bpgX = (outDelta.layerSize + TPB_X - 1) / TPB_X;
  int bpgY = (outDelta.batchSize + TPB_Y - 1) / TPB_Y;

  unsigned spitch = (TPB_X + 1);
  size_t sharedMemSize = 2 * spitch * TPB_Y * sizeof(float);

  backwardDeltaKernel<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1), sharedMemSize, stream>>>(
      nextDelta, transposedWeights, layerOutput, outDelta, spitch);
}





void GradientKernel::Apply(LayerBatchDeltas layerDeltas, LayerBatchOutputs layerOutputs,
                           LayerWeights outGradient, cudaStream_t stream) {

  assert(layerDeltas.batchSize == layerOutputs.batchSize);
  assert(layerDeltas.layerSize == outGradient.layerSize);
  assert(layerOutputs.layerSize == outGradient.inputSize);

  int bpgX = (outGradient.inputSize + TPB_X - 1) / TPB_X;
  int bpgY = (outGradient.layerSize + TPB_Y - 1) / TPB_Y;

  unsigned spitch = (TPB_X + 1);
  size_t sharedMemSize = 2 * spitch * TPB_Y * sizeof(float);

  gradientKernel<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1), sharedMemSize, stream>>>(
      layerDeltas, layerOutputs, outGradient, spitch);
}




void SoftmaxKernel::Apply(const LayerBatchOutputs &lastLayer, cudaStream_t stream) {
  size_t sharedMemSize = (lastLayer.layerSize + 1) * sizeof(float);

  // We dont want to include the bias part of the output in the processing of the softmax.
  int tpb = lastLayer.layerSize - 1;
  int bpg = lastLayer.batchSize;
  softmaxKernel<<<bpg, tpb, sharedMemSize, stream>>>(lastLayer);
}






using namespace std;

// ADAM trainer parameters
static constexpr float adamBeta1 = 0.9f;
static constexpr float adamBeta2 = 0.999f;
static constexpr float adamEpsilon = 10e-8;
static constexpr float adamLearnRate = 0.001f;

static Random rnd;
static std::once_flag stateFlag;

static void initialiseSharedState(void) {
  std::call_once(stateFlag, [](){
    rnd = Random::Create(2048, 1337);
  });
}

__global__ void initialiseLayerWeights(LayerWeights layer, const float initRange, Random rnd) {
  const unsigned row = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row >= layer.layerSize || col >= layer.inputSize) {
    return;
  }

  float *out = layer.Elem(row, col);
  *out = initRange * (rnd.SampleUniform(col + row * layer.inputSize) * 2.0f - 1.0f);
}

__global__ void initialiseLayerOutputs(LayerBatchOutputs outputs) {
  const unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= outputs.maxBatchSize) {
    return;
  }

  *(outputs.OutputElem(id, outputs.layerSize - 1)) = 1.0f;
}

__global__ void initialiseAdamWeights(LayerWeights momentum, LayerWeights rms) {
  assert(momentum.inputSize == rms.inputSize);
  assert(momentum.layerSize == rms.layerSize);

  const unsigned row = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row >= rms.layerSize || col >= rms.inputSize) {
    return;
  }

  *momentum.Elem(row, col) = 0.0f;
  *rms.Elem(row, col) = 0.0f;
}

__global__ void lastLayerDeltasKernel(LayerBatchOutputs networkOutput, SamplesBatch samples,
                                      LayerBatchDeltas out) {
  assert(networkOutput.layerSize == samples.targetOutputDim + 1);
  assert(out.layerSize == samples.targetOutputDim);

  const unsigned row = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row >= out.batchSize || col >= out.layerSize) {
    return;
  }

  // TODO: check whether reading into shared mem, doing computation, then writing to global mem
  // is faster. You never know.
  *out.Elem(row, col) = *networkOutput.OutputElem(row, col) - *samples.TargetOutputElem(row, col);
}

__global__ void updateMomentumAndRMS(LayerWeights gradient, LayerWeights momentum, LayerWeights rms,
                                      const float beta1, const float beta2) {
  const unsigned row = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row >= gradient.layerSize || col >= gradient.inputSize) {
    return;
  }

  float g = *gradient.Elem(row, col);
  float m = *momentum.Elem(row, col);
  float r = *rms.Elem(row, col);

  *momentum.Elem(row, col) = m * beta1 + g * (1.0f - beta1);
  *rms.Elem(row, col) = r * beta2 + g * g * (1.0f - beta2);
}

__global__ void updateWeightsWithAdam(LayerWeights weights, LayerWeights momentum, LayerWeights rms,
                                      const float beta1, const float beta2,
                                      const float lr, const float epsilon) {

  const unsigned row = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row >= rms.layerSize || col >= rms.inputSize) {
    return;
  }

  float mc = *momentum.Elem(row, col) / (1.0f - beta1);
  float rc = *rms.Elem(row, col) / (1.0f - beta2);

  *weights.Elem(row, col) -= lr * mc / sqrtf(rc + epsilon);
}

struct CudaNetwork::CudaNetworkImpl {
  NetworkSpec networkSpec;
  vector<LayerWeights> d_layerWeights;
  vector<LayerWeights> d_layerWeightsBridge;
  vector<LayerWeights> d_layerGradients;
  vector<LayerBatchOutputs> d_layerOutputs;
  vector<LayerBatchDeltas> d_layerDeltas;
  SamplesBatch d_samplesBatch;

  LayerWeights d_transposeScratch;

  // TODO: this stuff should go into a separate file. Trainer code/variables should be
  // separate from network code.
  vector<LayerWeights> d_adamMomentum;
  vector<LayerWeights> d_adamRMS;

  cudaStream_t uploadStream;
  cudaStream_t computeStream;

  CudaNetworkImpl(const NetworkSpec &spec) : networkSpec(spec) {
    assert(networkSpec.hiddenActivation != LayerActivation::SOFTMAX);
    initialiseSharedState();

    uploadStream = 0;
    computeStream = 0;

    allocDeviceMemory();
    initialiseWeights();
    initialiseOutputs();
    initialiseADAM();
  }

  ~CudaNetworkImpl() {
    for (auto& lw : d_layerWeights) { util::DeleteLayerWeights(lw); }
    for (auto& lw : d_layerWeightsBridge) { util::DeleteLayerWeights(lw); }
    for (auto& lg : d_layerGradients) { util::DeleteLayerWeights(lg); }
    for (auto& lo : d_layerOutputs) { util::DeleteLayerBatchOutputs(lo); }
    for (auto& ld : d_layerDeltas) { util::DeleteLayerBatchDeltas(ld); }
    for (auto& am : d_adamMomentum) { util::DeleteLayerWeights(am); }
    for (auto& am : d_adamRMS) { util::DeleteLayerWeights(am); }
    util::DeleteSamplesBatch(d_samplesBatch);
    util::DeleteLayerWeights(d_transposeScratch);
  }

  void SetWeights(const std::vector<math::MatrixView> &weights) {
    assert(d_layerWeights.size() == weights.size());

    for (unsigned i = 0; i < weights.size(); i++) {
      assert(weights[i].rows == d_layerWeights[i].layerSize);
      assert(weights[i].cols == d_layerWeights[i].inputSize);

      cudaError_t err = cudaMemcpy2D(
          d_layerWeights[i].weights, d_layerWeights[i].pitch,
          weights[i].data, weights[i].cols * sizeof(float),
          weights[i].cols * sizeof(float), weights[i].rows,
          cudaMemcpyHostToDevice);

      CheckError(err);
    }
  }

  void GetWeights(std::vector<math::MatrixView> &outWeights) {
    assert(outWeights.size() == d_layerWeightsBridge.size());

    for (unsigned i = 0; i < outWeights.size(); i++) {
      assert(outWeights[i].rows == d_layerWeightsBridge[i].layerSize);
      assert(outWeights[i].cols == d_layerWeightsBridge[i].inputSize);

      cudaError_t err = cudaMemcpy2DAsync(
          outWeights[i].data, outWeights[i].cols * sizeof(float), // dst
          d_layerWeightsBridge[i].weights, d_layerWeightsBridge[i].pitch, // src
          outWeights[i].cols * sizeof(float), outWeights[i].rows, // width, height
          cudaMemcpyDeviceToHost, uploadStream);

      CheckError(err);
    }
  }

  void Train(const math::MatrixView &batchInputs, const math::MatrixView &batchOutputs) {
    uploadSamplesBatch(batchInputs, batchOutputs);

    forwardPass();
    backwardPass();
    updateAdamParams();
    updateWeights();

    for (unsigned i = 0; i < d_layerWeights.size(); i++) {
      cudaError_t err = cudaMemcpy2D(
          d_layerWeightsBridge[i].weights, d_layerWeightsBridge[i].pitch,
          d_layerWeights[i].weights, d_layerWeights[i].pitch,
          d_layerWeights[i].inputSize * sizeof(float), d_layerWeights[i].layerSize,
          cudaMemcpyDeviceToDevice);

      CheckError(err);
    }
  }

private:
  void uploadSamplesBatch(const math::MatrixView &batchInputs,
                          const math::MatrixView &batchOutputs) {
    assert(batchInputs.rows == batchOutputs.rows);
    assert(batchInputs.rows <= d_samplesBatch.maxBatchSize);
    assert(batchInputs.cols == d_samplesBatch.inputDim);
    assert(batchOutputs.cols == d_samplesBatch.targetOutputDim);

    d_samplesBatch.batchSize = batchInputs.rows;

    cudaError_t err = cudaMemcpy2D(
        d_samplesBatch.input, d_samplesBatch.ipitch, // dst
        batchInputs.data, batchInputs.cols * sizeof(float), // src
        batchInputs.cols * sizeof(float), batchInputs.rows, // width, height
        cudaMemcpyHostToDevice);
    CheckError(err);

    err = cudaMemcpy2D(
        d_samplesBatch.targetOutput, d_samplesBatch.opitch, // dst
        batchOutputs.data, batchOutputs.cols * sizeof(float), // src
        batchOutputs.cols * sizeof(float), batchOutputs.rows, // width, height
        cudaMemcpyHostToDevice);
    CheckError(err);
  }

  void forwardPass(void) {
    for (auto& lo : d_layerOutputs) {
      lo.batchSize = d_samplesBatch.batchSize;
    }

    // copy the batch inputs into the first layer outputs.
    cudaError_t err = cudaMemcpy2DAsync(
        d_layerOutputs[0].output, d_layerOutputs[0].opitch, // dst
        d_samplesBatch.input, d_samplesBatch.ipitch,        // src
        d_samplesBatch.inputDim * sizeof(float), d_samplesBatch.batchSize, // width, height
        cudaMemcpyDeviceToDevice, computeStream);
    CheckError(err);

    for (unsigned i = 1; i < d_layerOutputs.size(); i++) {
      LayerActivation activation = (i == d_layerOutputs.size() - 1) ?
          networkSpec.outputActivation : networkSpec.hiddenActivation;

      ForwardPassKernel::Apply(d_layerWeights[i-1], d_layerOutputs[i-1], d_layerOutputs[i],
          activation, rnd, networkSpec.nodeActivationRate, i == (d_layerOutputs.size() - 1),
          computeStream);
    }

    LayerBatchOutputs lastLayer = d_layerOutputs[d_layerOutputs.size() - 1];
    if (networkSpec.outputActivation == LayerActivation::SOFTMAX) {
      SoftmaxKernel::Apply(lastLayer, computeStream);
    }
  }

  void backwardPass(void) {
    generateLayerDeltas();
    generateGradient();
  }

  void generateLayerDeltas(void) {
    for (auto& ld : d_layerDeltas) {
      ld.batchSize = d_samplesBatch.batchSize;
    }

    LayerBatchDeltas lastLayerDeltas = d_layerDeltas[d_layerDeltas.size() - 1];
    LayerBatchOutputs networkOutput = d_layerOutputs[d_layerOutputs.size() - 1];

    int bpgX = (lastLayerDeltas.layerSize + TPB_X - 1) / TPB_X;
    int bpgY = (lastLayerDeltas.batchSize + TPB_Y - 1) / TPB_Y;

    lastLayerDeltasKernel<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1)>>>(
        networkOutput, d_samplesBatch, lastLayerDeltas);

    for (int i = d_layerDeltas.size() - 2; i >= 0; i--) {
      LayerWeights transposedWeights;
      transposedWeights.inputSize = d_layerWeights[i + 1].layerSize;
      transposedWeights.layerSize = d_layerWeights[i + 1].inputSize;
      transposedWeights.weights = d_transposeScratch.weights;
      transposedWeights.pitch = d_transposeScratch.pitch;

      TransposeKernel::Apply(d_layerWeights[i + 1], transposedWeights, computeStream);

      BackwardDeltaKernel::Apply(d_layerDeltas[i + 1], transposedWeights, d_layerOutputs[i+1],
                                 d_layerDeltas[i], computeStream);
    }
  }

  void generateGradient(void) {
    for (unsigned i = 0; i < d_layerWeights.size(); i++) {
      GradientKernel::Apply(d_layerDeltas[i], d_layerOutputs[i], d_layerGradients[i], computeStream);
    }
  }

  void updateAdamParams(void) {
    for (unsigned i = 0; i < d_layerGradients.size(); i++) {
      int bpgX = (d_layerGradients[i].inputSize + TPB_X - 1) / TPB_X;
      int bpgY = (d_layerGradients[i].layerSize + TPB_Y - 1) / TPB_Y;

      updateMomentumAndRMS<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1)>>>(
          d_layerGradients[i], d_adamMomentum[i], d_adamRMS[i], adamBeta1, adamBeta2);
    }
  }

  void updateWeights(void) {
    for (unsigned i = 0; i < d_layerWeights.size(); i++) {
      int bpgX = (d_layerWeights[i].inputSize + TPB_X - 1) / TPB_X;
      int bpgY = (d_layerWeights[i].layerSize + TPB_Y - 1) / TPB_Y;

      updateWeightsWithAdam<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1)>>>(
          d_layerWeights[i], d_adamMomentum[i], d_adamRMS[i],
          adamBeta1, adamBeta2, adamLearnRate, adamEpsilon);
    }
  }

  void initialiseADAM(void) {
    assert(d_adamRMS.size() == d_adamMomentum.size());

    for (unsigned i = 0; i < d_adamRMS.size(); i++) {
      int bpgX = (d_adamRMS[i].inputSize + TPB_X - 1) / TPB_X;
      int bpgY = (d_adamRMS[i].layerSize + TPB_Y - 1) / TPB_Y;

      initialiseAdamWeights<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1)>>>(
          d_adamMomentum[i], d_adamRMS[i]);
    }
  }

  void initialiseOutputs(void) {
    // We initialise the outputs array for each layer to have a 1.0 at the end so that it can
    // be used as the bias input for the next layer.
    for (auto& lo : d_layerOutputs) {
      int bpgX = (lo.maxBatchSize + TPB_X - 1) / TPB_X;
      initialiseLayerOutputs<<<bpgX, TPB_X>>>(lo);
    }
  }

  void initialiseWeights(void) {
    for (auto& lw : d_layerWeights) {
      // Blocks per grid in X and Y dimensions.
      int bpgX = (lw.inputSize + TPB_X - 1) / TPB_X;
      int bpgY = (lw.layerSize + TPB_Y - 1) / TPB_Y;

      float initRange = 1.0f / sqrtf(lw.inputSize);
      initialiseLayerWeights<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1)>>>(lw, initRange, rnd);
    }
  }

  // Pre-allocated all of the device memory we will need. We should never have to malloc device
  // memory after this function is called.
  void allocDeviceMemory(void) {
    vector<unsigned> layerSizes(networkSpec.hiddenLayers.size() + 1);
    for (unsigned i = 0; i < networkSpec.hiddenLayers.size(); i++) {
      layerSizes[i] = networkSpec.hiddenLayers[i];
    }
    layerSizes[networkSpec.hiddenLayers.size()] = networkSpec.numOutputs;

    // This is for the input layer
    d_layerOutputs.push_back(
        util::NewLayerBatchOutputs(networkSpec.maxBatchSize, networkSpec.numInputs + 1));

    unsigned maxInputSize = 0;
    unsigned maxLayerSize = 0;

    for (unsigned i = 0; i < layerSizes.size(); i++) {
      unsigned prevLayerSize = i == 0 ? networkSpec.numInputs : layerSizes[i-1];

      maxInputSize = max(maxInputSize, prevLayerSize + 1);
      maxLayerSize = max(maxLayerSize, layerSizes[i]);

      d_layerWeights.push_back(util::NewLayerWeights(prevLayerSize + 1, layerSizes[i]));
      d_layerWeightsBridge.push_back(util::NewLayerWeights(prevLayerSize + 1, layerSizes[i]));
      d_layerGradients.push_back(util::NewLayerWeights(prevLayerSize + 1, layerSizes[i]));
      d_layerOutputs.push_back(util::NewLayerBatchOutputs(networkSpec.maxBatchSize, layerSizes[i] + 1));
      d_layerDeltas.push_back(util::NewLayerBatchDeltas(networkSpec.maxBatchSize, layerSizes[i]));

      d_adamMomentum.push_back(util::NewLayerWeights(prevLayerSize + 1, layerSizes[i]));
      d_adamRMS.push_back(util::NewLayerWeights(prevLayerSize + 1, layerSizes[i]));
    }

    d_samplesBatch =
        util::NewSamplesBatch(networkSpec.maxBatchSize, networkSpec.numInputs, networkSpec.numOutputs);

    d_transposeScratch = util::NewLayerWeights(maxLayerSize, maxInputSize);
  }
};


CudaNetwork::CudaNetwork(const NetworkSpec &spec) : impl(new CudaNetworkImpl(spec)) {}
CudaNetwork::~CudaNetwork() = default;

void CudaNetwork::SetWeights(const std::vector<math::MatrixView> &weights) {
    impl->SetWeights(weights);
}

void CudaNetwork::GetWeights(std::vector<math::MatrixView> &outWeights) {
  impl->GetWeights(outWeights);
}

void CudaNetwork::Train(const math::MatrixView &batchInputs, const math::MatrixView &batchOutputs) {
impl->Train(batchInputs, batchOutputs);
}



int main(){
	cout<<"HI \n"<<endl;
}



