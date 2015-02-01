#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <string.h> //for memset

#include "defines.h"

// When summing the outputs at a specific frame for each partial, we use a reduction method.
// This reduction method requires a temporary array in shared memory.
__shared__ float partialReductionOutputs[NUM_PARTIALS*NUM_CH];

__device__ __host__ void reduceOutputs(float *buffer, unsigned partialIdx, int sampleIdx, float outputL, float outputR) {
	//algorithm: given 8 outputs, [0, 1, 2, 3, 4, 5, 6, 7]
	//first iteration: 4 active threads. 
	//  Thread 0 adds i0 to i(0+4). Thread 1 adds i1 to i(1+4). Thread 2 adds i2 to i(2+4). Thread 3 adds i3 to i(3+4)
	//  Output now: [4, 6, 8, 10,   4, 5, 6, 7]
	//second iteration: 2 active threads.
	//  Thread 0 adds i0 to i(0+2). Thread 1 adds i1 to i(1+2)
	//  Output now: [12, 16,   8, 10, 4, 5, 6, 7]
	//third iteration: 1 active thread.
	//  Thread 0 adds i0 to i(0+1).
	//  Output now: [28,   16, 8, 10, 4, 5, 6, 7]
	//fourth iteration: 0 active threads -> exit
	#ifdef __CUDA_ARCH__
		//device code
		partialReductionOutputs[partialIdx * 2 + 0] = outputL;
		partialReductionOutputs[partialIdx * 2 + 1] = outputR;
		unsigned numActiveThreads = NUM_PARTIALS / 2;
		while (numActiveThreads > 0) {
			__syncthreads();
			partialReductionOutputs[partialIdx] += partialReductionOutputs[partialIdx + numActiveThreads];
			numActiveThreads /= 2;
		}
		if (partialIdx == 0) {
			buffer[sampleIdx * 2 + 0] = partialReductionOutputs[0];
			buffer[sampleIdx * 2 + 1] = partialReductionOutputs[1];
		}
	#else
		//host code
		//Since everything's computed iteratively, we can just add our outputs directly to the buffer.
		//First write to this sample must zero-initialize the buffer (not required in the GPU code).
		if (partialIdx == 0) {
			buffer[sampleIdx * 2 + 0] = 0;
			buffer[sampleIdx * 2 + 1] = 0;
		}
		buffer[sampleIdx * 2 + 0] += outputL;
		buffer[sampleIdx * 2 + 1] += outputR;
	#endif
}


__device__ __host__ void computePartialOutput(float *buffer, unsigned baseIdx, unsigned partialIdx, float fundamentalFreq) {
	float angleDelta = fundamentalFreq * INV_SAMPLE_RATE * (partialIdx+1);
	for (int sampleIdx = 0; sampleIdx < BUFFER_BLOCK_SIZE; ++sampleIdx) {
		float outputL, outputR;
		outputL = outputR = (1.0/NUM_PARTIALS)*sinf((baseIdx + sampleIdx) * angleDelta);
		reduceOutputs(buffer, partialIdx, sampleIdx, outputL, outputR);
	}
}

__global__ void fillSineWaveKernel(float *buffer, unsigned baseIdx, float fundamentalFreq) {
	int partialNum = threadIdx.x;
	computePartialOutput(buffer, baseIdx, partialNum, fundamentalFreq);
}

__host__ void fillSineWaveOnCpu(float *buffer, unsigned baseIdx, float fundamentalFreq) {
	for (int partialIdx = 0; partialIdx < NUM_PARTIALS; ++partialIdx) {
		computePartialOutput(buffer, baseIdx, partialIdx, fundamentalFreq);
	}
}

__host__ void fillSineWaveCuda(float *bufferB, unsigned baseIdx, float fundamentalFreq) {
	float *gpuOutBuff;
	//allocate buffer memory. No need for it to be zero'd
	cudaMalloc(&gpuOutBuff,    BUFFER_BLOCK_SIZE*NUM_CH*sizeof(float));
	fillSineWaveKernel << <1, NUM_PARTIALS >> >(gpuOutBuff, baseIdx, fundamentalFreq);
	//copy memory into the cpu buffer
	//Note: this will wait for the kernel to complete first.
	cudaMemcpy(bufferB, gpuOutBuff, BUFFER_BLOCK_SIZE*NUM_CH*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(gpuOutBuff);
}

void fillSineWaveVoice(float *bufferB, unsigned baseIdx, float fundamentalFreq) {
	#if USE_CUDA
		fillSineWaveCuda(bufferB, baseIdx, fundamentalFreq);
	#else
		fillSineWaveOnCpu(bufferB, baseIdx, fundamentalFreq);
	#endif
}
