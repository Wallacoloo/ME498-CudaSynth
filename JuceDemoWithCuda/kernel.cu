#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

#include "defines.h"

__global__ void fillSineWaveKernel(float *buffer, unsigned baseIdx, float level, float angleDelta) {
	int bufIdx = threadIdx.x;
	buffer[bufIdx] = level*sin((baseIdx + bufIdx) * angleDelta);
}

void cudaFillSineWaveVoice(float *bufferB, unsigned baseIdx, float level, float angleDelta) {
	float *gpuOutBuff;
	cudaMalloc(&gpuOutBuff, BUFFER_BLOCK_SIZE*sizeof(float));
	fillSineWaveKernel<<<1, BUFFER_BLOCK_SIZE >>>(gpuOutBuff, baseIdx, level, angleDelta);
	//copy memory into the cpu buffer
	cudaMemcpy(bufferB, gpuOutBuff, BUFFER_BLOCK_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(gpuOutBuff);
}
