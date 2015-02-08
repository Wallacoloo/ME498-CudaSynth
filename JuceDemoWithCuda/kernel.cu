#include "kernel.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <string.h> //for memset
#include <assert.h>
#include <stdlib.h> //for atexit
#include <mutex>
#include <thread> //for unique_lock

#include "defines.h"

#define CIRCULAR_BUFFER_LEN MAX_DELAY_EFFECT_LENGTH

namespace kernel {
	// Must define our own generic complex class to run on both CPU and GPU.
	// std::complex does not have __device__  defines, meaning it will only generate host code.
	template <typename F> class ComplexT {
		F _r, _i;
	public:
		__device__ __host__ ComplexT() : _r(0), _i(0) {}
		__device__ __host__ ComplexT(F real, F imag) : _r(real), _i(imag) {}
		__device__ __host__ F real() const {
			return _r;
		}
		__device__ __host__ F imag() const {
			return _i;
		}
		__device__ __host__ ComplexT operator+(const ComplexT &other) const {
			return ComplexT(_r + other._r, _i + other._i);
		}
		__device__ __host__ ComplexT& operator+=(const ComplexT &other) {
			return (*this = (*this + other));
		}
		__device__ __host__ ComplexT operator*(const ComplexT &other) const {
			//(a+bi)(c+di) = ac + adi + bci + bd(-1)
			// = (ac-bd) + (ad+bc)i;
			return ComplexT(_r*other._r - _i*other._i, _r*other._i + _i*other._r);
		}
		__device__ __host__ ComplexT& operator*=(const ComplexT &other) {
			return (*this = (*this * other));
		}
	};

	// Use complex float pairs to represent the phase functions
	typedef ComplexT<float> PhaseT;

	// Efficient way to compute sequential ADSR values
	class ADSRState {
		enum Mode {
			AttackMode,
			DecayMode,
			SustainMode,
			ReleaseMode
		};
		Mode mode;
		float value;
		// in attack mode, we increase value by dv/dt, if the attack is constant.
		// but if attack is changing, then we want to interpolate the dv1/dt and dv2/dt
		// Thus, we have dv/dt = dv1/dt + (dv2/dt-dv1/dt)*t
		// Or, each sample, dv/dt += (dv2/dt-dv1/dt)
		float attackPrime;
		float attackDoublePrime;
	public:
		// call at block begin to precompute values.
		__device__ __host__ void atBlockStart(ADSR *start, ADSR *end, unsigned partialIdx) {
			value = 1.0;
			float dv_dtInSecondsAtAttack = 1.0 / start->getAttack();
			attackPrime = INV_SAMPLE_RATE * dv_dtInSecondsAtAttack;
			float dv_dtInSecondsAtAttack2 = 1.0 / end->getAttack();
			float dv_dt2_minus_dv_dt1InSeconds = dv_dtInSecondsAtAttack2 - dv_dtInSecondsAtAttack;
			attackDoublePrime = INV_SAMPLE_RATE * dv_dt2_minus_dv_dt1InSeconds;
		}
		__device__ __host__ float next() {
			switch (mode) {
			case AttackMode:
				attackPrime += attackDoublePrime;
				value += attackPrime;
				if (value >= 1.0f) {
					value = 1.0f;
					mode = DecayMode;
				}
				break;
			case DecayMode:
				break;
			default:
			case SustainMode:
				break;
			case ReleaseMode:
				break;
			}
			return value;
		}
	};

	// Contains info about the parameter states at ANY sample in the block
	struct FullBlockParameterInfo {
		ParameterStates start;
		ParameterStates end;
	};

	// Contains extra state information relevant to each individual partial
	struct PartialState {
		// The partial has a phase function, phase(t).
		// For constant frequency, phase(t) = w*t.
		// We need varied frequency over time whenever the frequency changes.
		// Thus, dp/dt = w0 + (w1-w0)/T*t, where T is the time over which the frequency should be altered.
		// Write as dp/dt = w0 + kt
		// and so each sample, the phase accumulator should be multiplied by e^i(w0+kt)
		// This *can* be done efficiently.
		// First, evaluate e^iw0 at the start of the block, store as dP/dt
		//   Also evaluate e^ik(1) as d^2P/dt^2. Each sample, multiply dP/dt = dP/dt * d^2P/dt^2
		PhaseT phase;
		PhaseT phasePrime;
		PhaseT phaseDoublePrime;
		ADSRState volumeEnvelope;
		PartialState() {}
		PartialState(struct SynthState *d_synthState, unsigned voiceNum, unsigned partialIdx)
		  : phase(1, 0) {
		}
	};

	struct SynthVoiceState {
		FullBlockParameterInfo parameterInfo;
		PartialState partialStates[NUM_PARTIALS];
	};

	// Packages all the state-related information for the synth in one class to store persistently on the device
	struct SynthState {
		SynthVoiceState voiceStates[MAX_SIMULTANEOUS_SYNTH_NOTES];
		float sampleBuffer[CIRCULAR_BUFFER_LEN*NUM_CH];
	};

	// When summing the outputs at a specific frame for each partial, we use a reduction method.
	// This reduction method requires a temporary array in shared memory.
	__shared__ float partialReductionOutputs[NUM_PARTIALS*NUM_CH];

	// this is a circular buffer of sample data (interleaved by channel number) stored on the device
	// It is persistent and lengthy, in order to accomodate the delay effect.
	SynthState *d_synthState = NULL;

	// When running on the cpu, we need to control concurrent access to the synth state
	std::mutex synthStateMutex;


	static void checkCudaError(cudaError_t e) {
		if (e != cudaSuccess) {
			printf("Cuda Error: %s\n", cudaGetErrorString(e));
			printf("Aborting\n");
			exit(1);
		}
	}

	static bool _hasCudaDevice() {
		int deviceCount;
		cudaError_t err = cudaGetDeviceCount(&deviceCount);
		// if we get a cuda error, it may be because the system has no cuda dlls.
		bool useCuda = (err == cudaSuccess && deviceCount != 0);
		printf("Using Cuda? %i\n", useCuda);
		return useCuda;
	}

	static bool hasCudaDevice() {
		//only check for the presence of a device once.
		static bool hasDevice = _hasCudaDevice();
		return hasDevice;
	}

	// code to run at shutdown (free buffers, etc)
	static void teardown() {
		// free the sample buffer if we allocated it and it hasn't already been freed.
		if (d_synthState != NULL) {
			if (hasCudaDevice()) {
				checkCudaError(cudaFree(d_synthState));
			}
			else {
				free(d_synthState);
			}
			// avoid double-frees
			d_synthState = NULL;
		}
	}

	// code to run on first-time audio calculation
	static void startup() {
		atexit(&teardown);
		if (hasCudaDevice()) {
			// allocate sample buffer on device
			checkCudaError(cudaMalloc(&d_synthState, sizeof(SynthState)));
		} else {
			// allocate sample buffer on cpu
			d_synthState = (SynthState*)malloc(sizeof(SynthState));
			printf("Allocated synthState %p\n", d_synthState);
		}
	}

	static void doStartupOnce() {
		static bool hasInit = false;
		if (!hasInit) {
			startup();
			hasInit = true;
		}
	}

	// called for each partial to sum their outputs together.
	__device__ __host__ void reduceOutputs(SynthState *synthState, unsigned partialIdx, int sampleIdx, float outputL, float outputR) {
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
		unsigned bufferIdx = NUM_CH * (sampleIdx % CIRCULAR_BUFFER_LEN);
#ifdef __CUDA_ARCH__
		//device code
		partialReductionOutputs[NUM_CH*partialIdx + 0] = outputL;
		partialReductionOutputs[NUM_CH*partialIdx + 1] = outputR;
		unsigned numActiveThreads = NUM_PARTIALS / 2;
		while (numActiveThreads > 0) {
			__syncthreads();
			partialReductionOutputs[NUM_CH*partialIdx + 0] += partialReductionOutputs[NUM_CH*partialIdx + numActiveThreads*NUM_CH + 0];
			partialReductionOutputs[NUM_CH*partialIdx + 1] += partialReductionOutputs[NUM_CH*partialIdx + numActiveThreads*NUM_CH + 1];
			numActiveThreads /= 2;
		}
		if (partialIdx == 0) {
			synthState->sampleBuffer[bufferIdx + 0] = partialReductionOutputs[0];
			synthState->sampleBuffer[bufferIdx + 1] = partialReductionOutputs[1];
		}
#else
		//host code
		//Since everything's computed iteratively, we can just add our outputs directly to the buffer.
		//First write to this sample must zero-initialize the buffer (not required in the GPU code).
		if (partialIdx == 0) {
			synthState->sampleBuffer[bufferIdx + 0] = 0;
			synthState->sampleBuffer[bufferIdx + 1] = 0;
		}
		synthState->sampleBuffer[bufferIdx + 0] += outputL;
		synthState->sampleBuffer[bufferIdx + 1] += outputR;
#endif
	}

	// called at the end of the block.
	// if parameterInfo.start != parameterInfo.end, then we copy the end parameters of this block to the start parameters for the next block.
	// this needs to be called for each sine wave.
	__device__ __host__ void updateVoiceParametersIfNeeded(SynthVoiceState *voiceState, unsigned voiceNum, unsigned partialIdx) {
		/*int transferSize = 16;
		int totalBytesToCopy = sizeof(ParameterStates);
		int numTransfers = (totalBytesToCopy + transferSize - 1) / transferSize;
		int numTransfersPerThread = (numTransfers + NUM_PARTIALS - 1) / NUM_PARTIALS;*/
		if (partialIdx == 0) {
			memcpy(&voiceState->parameterInfo.start, &voiceState->parameterInfo.end, sizeof(ParameterStates));
		}

	}

	// compute the output for ONE sine wave over the current block
	__device__ __host__ void computePartialOutput(SynthState *synthState, unsigned voiceNum, unsigned baseIdx, unsigned partialIdx, float fundamentalFreq) {
		SynthVoiceState *voiceState = &synthState->voiceStates[voiceNum];
		float angleDelta = fundamentalFreq * INV_SAMPLE_RATE * (partialIdx + 1);
		PartialState* myState = &voiceState->partialStates[partialIdx];
		myState->volumeEnvelope.atBlockStart(&voiceState->parameterInfo.start.volumeEnvelope, &voiceState->parameterInfo.end.volumeEnvelope, partialIdx);
		// myState->phase = PhaseT(1, 0);
		// compute e^iw0*deltaT.
		// = cos(w0*deltaT) + i*sin(w0*deltaT)
		myState->phasePrime = PhaseT(cosf(angleDelta), sinf(angleDelta));
		myState->phaseDoublePrime = PhaseT(1, 0);
		for (int sampleIdx = 0; sampleIdx < BUFFER_BLOCK_SIZE; ++sampleIdx) {
			float outputL, outputR;
			float sinusoid = myState->phase.imag(); // Extract the sinusoidal portion of the wave.
			myState->phasePrime *= myState->phaseDoublePrime;
			myState->phase *= myState->phasePrime;
			float amplitude = (1.0 / NUM_PARTIALS) * voiceState->parameterInfo.start.partialLevels[partialIdx];
			amplitude *= myState->volumeEnvelope.next();
			outputL = outputR = amplitude*sinusoid;
			reduceOutputs(synthState, partialIdx, baseIdx + sampleIdx, outputL, outputR);
			updateVoiceParametersIfNeeded(voiceState, voiceNum, partialIdx);
		}
	}

	__global__ void evaluateSynthVoiceBlockKernel(SynthState *synthState, unsigned voiceNum, unsigned baseIdx, float fundamentalFreq) {
		int partialNum = threadIdx.x;
		computePartialOutput(synthState, voiceNum, baseIdx, partialNum, fundamentalFreq);
	}

	__host__ void evaluateSynthVoiceBlockOnCpu(float *bufferB, unsigned voiceNum, unsigned sampleIdx, float fundamentalFreq) {
		// need to obtain a lock on the synth state
		std::unique_lock<std::mutex> stateLock(synthStateMutex);
		for (int partialIdx = 0; partialIdx < NUM_PARTIALS; ++partialIdx) {
			computePartialOutput(d_synthState, voiceNum, sampleIdx, partialIdx, fundamentalFreq);
		}
		unsigned bufferStartIdx = NUM_CH * (sampleIdx % CIRCULAR_BUFFER_LEN);
		memcpy(bufferB, &d_synthState->sampleBuffer[bufferStartIdx], BUFFER_BLOCK_SIZE*NUM_CH*sizeof(float));
	}

	__host__ void evaluateSynthVoiceBlockCuda(float *bufferB, unsigned voiceNum, unsigned sampleIdx, float fundamentalFreq) {
		// update the ending parameter states of this block
		// if (newParameters) {
		//	checkCudaError(cudaMemcpy(&d_synthState->parameterInfo.end, newParameters, sizeof(ParameterStates), cudaMemcpyHostToDevice));
		//}
		evaluateSynthVoiceBlockKernel << <1, NUM_PARTIALS >> >(d_synthState, voiceNum, sampleIdx, fundamentalFreq);

		checkCudaError(cudaGetLastError()); //check if error in kernel launch
		checkCudaError(cudaDeviceSynchronize()); //check for error INSIDE the kernel

		//copy memory into the cpu buffer
		//Note: this will wait for the kernel to complete first.
		unsigned bufferStartIdx = NUM_CH * (sampleIdx % CIRCULAR_BUFFER_LEN);
		checkCudaError(cudaMemcpy(bufferB, &d_synthState->sampleBuffer[bufferStartIdx], BUFFER_BLOCK_SIZE*NUM_CH*sizeof(float), cudaMemcpyDeviceToHost));
	}

	void evaluateSynthVoiceBlock(float *bufferB, unsigned voiceNum, unsigned baseIdx, float fundamentalFreq) {
		doStartupOnce();
		if (hasCudaDevice()) {
			evaluateSynthVoiceBlockCuda(bufferB, voiceNum, baseIdx, fundamentalFreq);
		} else {
			evaluateSynthVoiceBlockOnCpu(bufferB, voiceNum, baseIdx, fundamentalFreq);
		}
	}

	static void memcpyHostToSynthState(void *dest, const void *src, std::size_t numBytes) {
		// if running on device, copy params to GPU via cudaMemcpy, else normal memcpy on cpu.
		if (hasCudaDevice()) {
			// cudaMemcpy is synchronous, so concurrency is dealt with automatically
			checkCudaError(cudaMemcpy(dest, src, numBytes, cudaMemcpyHostToDevice));
		}
		else {
			// else, copy them using normal memcpy
			// Must first obtain a lock to the synth data.
			std::unique_lock<std::mutex> stateLock(synthStateMutex);
			memcpy(dest, src, numBytes);
		}
	}

	static void copyParameterStates(const ParameterStates *newParameters, ParameterStates *dest) {
		memcpyHostToSynthState(dest, newParameters, sizeof(ParameterStates));
		// if running on device, copy params to GPU via cudaMemcpy, else normal memcpy on cpu.
		/*if (hasCudaDevice()) {
			// cudaMemcpy is synchronous, so concurrency is dealt with automatically
			checkCudaError(cudaMemcpy(dest, newParameters, sizeof(ParameterStates), cudaMemcpyHostToDevice));
		} else {
			// else, copy them using normal memcpy
			// Must first obtain a lock to the synth data.
			// TODO: Should only lock once inside the larger loop
			std::unique_lock<std::mutex> stateLock(synthStateMutex);
			memcpy(dest, newParameters, sizeof(ParameterStates));
		}*/
	}

	void parameterStatesChanged(const ParameterStates *newParameters) {
		doStartupOnce();
		static bool hasInitStartParams = false;

		for (int i = 0; i < MAX_SIMULTANEOUS_SYNTH_NOTES; ++i) {
			// If this is the first time we've received parameter states, then that means parameterInfo.start is uninitialized.
			if (!hasInitStartParams) {
				copyParameterStates(newParameters, &d_synthState->voiceStates[i].parameterInfo.start);
			}
			copyParameterStates(newParameters, &d_synthState->voiceStates[i].parameterInfo.end);
		}
		hasInitStartParams = true;
	}

	void onNoteStart(unsigned voiceNum) {
		doStartupOnce();
		// need to go through and properly initialize all the note's state information:
		//   partial phases, ADSR states, etc.
		PartialState partialStates[NUM_PARTIALS];
		for (int i = 0; i < NUM_PARTIALS; ++i) {
			partialStates[i] = PartialState(d_synthState, voiceNum, i);
		}
		memcpyHostToSynthState(&d_synthState->voiceStates[voiceNum].partialStates, partialStates, sizeof(PartialState)*NUM_PARTIALS);
	}

}