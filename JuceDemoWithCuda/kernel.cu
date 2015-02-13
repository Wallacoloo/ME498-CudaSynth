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
		__device__ __host__ ComplexT(F angleRad) {
			sincosf(angleRad, &_i, &_r);
		}
		__device__ __host__ F real() const {
			return _r;
		}
		__device__ __host__ F imag() const {
			return _i;
		}
		__device__ __host__ F magSq() const {
			return _r*_r + _i*_i;
		}
		__device__ __host__ F mag() const {
			return sqrtf(magSq());
		}
		__device__ __host__ F phase() const {
			return atan2(_i, _r);
		}
		__device__ __host__ ComplexT inverse() const {
			// return 1.0/(this)
			// 1 / (a+b*i) = (a-b*i) / (a^2+b^2)
			return ComplexT(_r, -_i) / magSq();
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
		__device__ __host__ ComplexT operator*(F other) const {
			return (*this) * ComplexT(other, 0);
		}
		__device__ __host__ ComplexT operator/(const ComplexT &other) const {
			return (*this) * other.inverse();
		}
		__device__ __host__ ComplexT operator/(F other) const {
			return (*this) * (1.f / other);
		}
		__device__ __host__ ComplexT& operator*=(const ComplexT &other) {
			return (*this = (*this * other));
		}
		__device__ __host__ ComplexT& operator*=(F other)  {
			return (*this) *= ComplexT(other, 0);
		}
		__device__ __host__ ComplexT& operator/=(const ComplexT &other) {
			return (*this) *= other.inverse();
		}
		__device__ __host__ ComplexT& operator/=(F other) {
			return (*this) *= (1.f / other);
		}
		__device__ __host__ ComplexT pow(F n) {
			//(r*e^(i*phase))^n = r^n*e^(i*n*phase)
			F newMag = powf(mag(), n);
			F newPhase = powf(phase(), n);
			return ComplexT(newPhase)*newMag;
		}
	};

	// Use complex float pairs to represent the phase functions
	typedef ComplexT<float> PhaseT;

	// Efficient way to compute successive sine values
	class Sinuisoidal {
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
	public:
		Sinuisoidal() : phase(1, 0) {}
		__device__ __host__ void newFrequency(float start, float end) {
			float wStart = INV_SAMPLE_RATE * start;
			float wEnd = INV_SAMPLE_RATE * end;
			PhaseT phasePrimeStart = PhaseT(wStart);
			PhaseT phasePrimeEnd = PhaseT(wEnd);
			phasePrime = phasePrimeStart;
			phaseDoublePrime = PhaseT(1, 0);
			// phasePrimeStart * doublePrime^BUFFER_BLOCK_SIZE = phasePrimeEnd
			// (phasePrimeEnd/phasePrimeStart)^(1.0/BUFFER_BLOCK_SIZE) = doublePrime
			// phaseDoublePrime = PhaseT(powf(phasePrimeEnd/phasePrimeStart, 1.0 / BUFFER_BLOCK_SIZE));
			// Note: (a+bi)^n = (r*e^(i*p))^n = r^n*e^(i*n*p)
			// phaseDoublePrime = (phasePrimeEnd / phasePrimeEnd).pow(1.0 / BUFFER_BLOCK_SIZE);
		}
		__device__ __host__ void newFrequency(float frequency) {
			newFrequency(frequency, frequency);
		}
		__device__ __host__ void newDepth(float depth) {
			// make the current magnitude match the desired depth
			// we must avoid the division by zero if the current depth is 0.
			// to avoid this, we just prevent the current depth from ever *being* zero. 
			// In this way, we also don't lose track of position when the depth is toggled to 0,
			// but there will always be some *small* component of LFO influencing things.
			depth = max(0.0001, depth);
			float mag = phase.mag();
			// float mag = max(0.0001, phase.mag());
			float factor = depth / mag;
			phase *= factor;
		}
		__device__ __host__ PhaseT next() {
			phasePrime *= phaseDoublePrime;
			phase *= phasePrime;
			return phase;
		}
	};

	// Efficient way to compute sequential ADSR values
	class ADSRState {
		enum Mode {
			AttackMode,
			DecayMode,
			SustainMode,
			ReleaseMode,
			EndMode,
		};
		Mode mode;
		float value;
		// attack approach #1 (makes more sense in the context where dv2/dt = infinity)
		// attackTime(t) = a1 + (a2-a1)*t/deltaT
		// dv/dt = 1.0/attackTime(t)
		// dv/dt = 1.0/(a1 + (a2-a1)*t/deltaT)
		// However, this is difficult to compute efficiently

		// attack approach #2
		// in attack mode, we increase value by dv/dt, if the attack is constant.
		// but if attack is changing, then we want to interpolate the dv1/dt and dv2/dt
		// Thus, we have dv/dt = dv1/dt + (dv2/dt-dv1/dt)*t
		// Or, each sample, dv/dt += (dv2/dt-dv1/dt)
		float attackPrime;
		float attackDoublePrime;

		// decay approach (a bit more difficult, since depends on changing sustain levels too).
		// calculate dv1/dt and dv2/dt just as in attack approach #2, based on the start decay&sustain and the end decay&sustain.
		// Then apply same algorithm.
		float decayPrime;
		float decayDoublePrime;

		// sustain approach
		// calculate sustain1 and sustain2.
		// then sustainLevel = sustain1 + (sustain2-sustain1)*t/deltaT
		float sustainLevel;
		float sustainPrime;

		// For release, use the same approach as the decay.
		float releasePrime;
		float releaseDoublePrime;
	public:
		// initialized at the start of a note
		ADSRState() : mode(AttackMode), value(0.f) {}
		// call at block begin to precompute values.
		__device__ __host__ void atBlockStart(ADSR *start, ADSR *end, unsigned partialIdx, bool released) {
			if (mode == EndMode) {
				return;
			}
			float startAttack =  start->getAttackFor(partialIdx);
			float startDecay =   start->getDecayFor(partialIdx);
			float startSustain = start->getSustain();
			float startRelease = start->getReleaseFor(partialIdx);
			float endAttack =    end->getAttackFor(partialIdx);
			float endDecay =     end->getDecayFor(partialIdx);
			float endSustain =   end->getSustain();
			float endRelease =   end->getReleaseFor(partialIdx);
			float beginReleaseLevelStart, beginReleaseLevelEnd;
			if (released) {
				mode = ReleaseMode;
				// release starts from current value
				beginReleaseLevelStart = value;
				beginReleaseLevelEnd = value;
			} else {
				// release will start from the sustain value
				beginReleaseLevelStart = startRelease;
				beginReleaseLevelEnd = endRelease;
			}
			// Calculate attack parameters
			float dv_dtInSecondsAtAttack = 1.f / max(startAttack, INV_SAMPLE_RATE);
			attackPrime = INV_SAMPLE_RATE * dv_dtInSecondsAtAttack;
			float dv_dtInSecondsAtAttack2 = 1.f / max(endAttack, INV_SAMPLE_RATE);
			float dv_dt2_minus_dv_dt1InSecondsAtAttack = dv_dtInSecondsAtAttack2 - dv_dtInSecondsAtAttack;
			attackDoublePrime = INV_SAMPLE_RATE * dv_dt2_minus_dv_dt1InSecondsAtAttack;
			// Calculate delay parameters
			float dv_dtInSecondsAtDecay = (startSustain-1.f) / max(startDecay, INV_SAMPLE_RATE);
			decayPrime = INV_SAMPLE_RATE * dv_dtInSecondsAtDecay;
			float dv_dtInSecondsAtDecay2 = (endSustain - 1.f) / max(endDecay, INV_SAMPLE_RATE);
			float dv_dt2_minus_dv_dt1InSecondsAtDecay = dv_dtInSecondsAtDecay2 - dv_dtInSecondsAtDecay;
			decayDoublePrime = INV_SAMPLE_RATE * dv_dt2_minus_dv_dt1InSecondsAtDecay;
			// Calculate sustain parameters
			sustainLevel = startSustain;
			sustainPrime = INV_BUFFER_BLOCK_SIZE * (endSustain - startSustain);
			// Calculate release parameters
			float dv_dtInSecondsAtRelease = (0.f - beginReleaseLevelStart) / max(startRelease, INV_SAMPLE_RATE);
			releasePrime = INV_SAMPLE_RATE * dv_dtInSecondsAtRelease;
			float dv_dtInSecondsAtRelease2 = (0.f - beginReleaseLevelEnd) / max(endRelease, INV_SAMPLE_RATE);
			float dv_dt2_minus_dv_dt1InSecondsAtRelease = dv_dtInSecondsAtRelease2 - dv_dtInSecondsAtRelease;
			releaseDoublePrime = INV_SAMPLE_RATE * dv_dt2_minus_dv_dt1InSecondsAtRelease;
		}
		__device__ __host__ bool isActive() const {
			return mode != EndMode;
		}
		__device__ __host__ float next() {
			sustainLevel += sustainPrime;
			switch (mode) {
			case AttackMode:
				value += attackPrime;
				attackPrime += attackDoublePrime;
				// check if it's time to move to the next mode or if value concave-down & no longer increasing
				if (value >= 1.0f || attackPrime < 0) {
					value = 1.0f;
					mode = DecayMode;
				}
				break;
			case DecayMode:
				value += decayPrime;
				decayPrime += decayDoublePrime;
				// check if it's time to move to the next mode or if value is concave-up & no longer decreasing
				if (value < sustainLevel || decayPrime > 0) {
					value = sustainLevel;
					mode = SustainMode;
				}
				break;
			case SustainMode:
				// must update value to the new sustain level computed above
				value = sustainLevel;
				break;
			case ReleaseMode:
				value += releasePrime;
				releasePrime += releaseDoublePrime;
				// check if it's time to move to the next mode or if value is concave-up & no longer decreasing
				if (value <= 0.f || releasePrime > 0) {
					value = 0.f;
					mode = EndMode;
				}
				break;
			default:
			case EndMode:
				break;
			}
			return value;
		}
	};

	class LFOState {
		Sinuisoidal sinusoid;
	public:
		__device__ __host__ void atBlockStart(LFO *start, LFO *end, unsigned partialIdx) {
			sinusoid.newFrequency(start->getLfoFreqFor(partialIdx));
			sinusoid.newDepth(start->getLfoDepthFor(partialIdx));
		}
		__device__ __host__ float next() {
			return sinusoid.next().imag();
		}
	};

	class ADSRLFOEnvelopeState {
		ADSRState adsr;
		LFOState lfo;
	public:
		__device__ __host__ void atBlockStart(ADSRLFOEnvelope *envStart, ADSRLFOEnvelope *envEnd, unsigned partialIdx, bool released) {
			adsr.atBlockStart(envStart->getAdsr(), envEnd->getAdsr(), partialIdx, released);
			lfo.atBlockStart(envStart->getLfo(), envEnd->getLfo(), partialIdx);
		}
		__device__ __host__ float next() {
			return adsr.next() * (1 + lfo.next());
		}
		__device__ __host__ bool isActive() const {
			return adsr.isActive();
		}
	};

	// Contains info about the parameter states at ANY sample in the block
	struct FullBlockParameterInfo {
		ParameterStates start;
		ParameterStates end;
	};

	// Contains extra state information relevant to each individual partial
	struct PartialState {
		Sinuisoidal sinusoid;
		ADSRLFOEnvelopeState volumeEnvelope;
		PartialState() {}
		PartialState(struct SynthState *synthState, unsigned voiceNum, unsigned partialIdx) {}
		__device__ __host__ void atBlockStart(struct SynthVoiceState *voiceState, unsigned partialIdx, float fundamentalFreq, bool released);
	};

	struct SynthVoiceState {
		FullBlockParameterInfo parameterInfo;
		PartialState partialStates[NUM_PARTIALS];
		float sampleBuffer[CIRCULAR_BUFFER_LEN*NUM_CH];
	};

	// Packages all the state-related information for the synth in one class to store persistently on the device
	struct SynthState {
		SynthVoiceState voiceStates[MAX_SIMULTANEOUS_SYNTH_NOTES];
	};

	void PartialState::atBlockStart(struct SynthVoiceState *voiceState, unsigned partialIdx, float fundamentalFreq, bool released) {
		sinusoid.newFrequency((partialIdx + 1)*fundamentalFreq);
		volumeEnvelope.atBlockStart(&voiceState->parameterInfo.start.volumeEnvelope, &voiceState->parameterInfo.end.volumeEnvelope, partialIdx, released);
	}

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
		std::unique_lock<std::mutex> stateLock(synthStateMutex);
		// free the sample buffer if we allocated it and it hasn't already been freed.
		if (d_synthState != NULL) {
			if (hasCudaDevice()) {
				checkCudaError(cudaFree(d_synthState));
			} else {
				free(d_synthState);
			}
			// avoid double-frees
			d_synthState = NULL;
		}
	}

	// code to run on first-time audio calculation
	static void startup() {
		atexit(&teardown);
		//SynthState defaultState;
		std::unique_lock<std::mutex> stateLock(synthStateMutex);
		if (hasCudaDevice()) {
			// allocate sample buffer on device
			checkCudaError(cudaMalloc(&d_synthState, sizeof(SynthState)));
			checkCudaError(cudaMemset(d_synthState, 0, sizeof(SynthState)));
			//checkCudaError(cudaMemcpy(d_synthState, &defaultState, sizeof(SynthState), cudaMemcpyHostToDevice));
		} else {
			// allocate sample buffer on cpu
			d_synthState = (SynthState*)malloc(sizeof(SynthState));
			memset(d_synthState, 0, sizeof(SynthState));
			//memcpy(d_synthState, &defaultState, sizeof(SynthState));
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
	__device__ __host__ void reduceOutputs(SynthVoiceState *voiceState, unsigned partialIdx, int sampleIdx, float outputL, float outputR) {
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
			voiceState->sampleBuffer[bufferIdx + 0] = partialReductionOutputs[0];
			voiceState->sampleBuffer[bufferIdx + 1] = partialReductionOutputs[1];
		}
#else
		//host code
		//Since everything's computed iteratively, we can just add our outputs directly to the buffer.
		//First write to this sample must zero-initialize the buffer (not required in the GPU code).
		if (partialIdx == 0) {
			voiceState->sampleBuffer[bufferIdx + 0] = 0;
			voiceState->sampleBuffer[bufferIdx + 1] = 0;
		}
		voiceState->sampleBuffer[bufferIdx + 0] += outputL;
		voiceState->sampleBuffer[bufferIdx + 1] += outputR;
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
			// TODO: only do this copy if the parameters have changed
			memcpy(&voiceState->parameterInfo.start, &voiceState->parameterInfo.end, sizeof(ParameterStates));
		}

	}

	// compute the output for ONE sine wave over the current block
	__device__ __host__ void computePartialOutput(SynthState *synthState, unsigned voiceNum, unsigned baseIdx, unsigned partialIdx, float fundamentalFreq, bool released) {
		SynthVoiceState *voiceState = &synthState->voiceStates[voiceNum];
		PartialState* myState = &voiceState->partialStates[partialIdx];
		myState->atBlockStart(voiceState, partialIdx, fundamentalFreq, released);
		for (int sampleIdx = 0; sampleIdx < BUFFER_BLOCK_SIZE; ++sampleIdx) {
			float outputL, outputR;
			// Extract the sinusoidal portion of the wave.
			float sinusoid = myState->sinusoid.next().imag();
			// Get the base partial level (the hand-drawn frequency weights)
			float level = (1.0 / NUM_PARTIALS) * voiceState->parameterInfo.start.partialLevels[partialIdx];
			// Get the ADSR/LFO volume envelope
			float envelope = myState->volumeEnvelope.next();
			outputL = outputR = level*envelope*sinusoid;

			reduceOutputs(voiceState, partialIdx, baseIdx + sampleIdx, outputL, outputR);
			updateVoiceParametersIfNeeded(voiceState, voiceNum, partialIdx);
		}
		// TODO: use a proper reduction algorithm to determine when the note is complete
		if (partialIdx == NUM_PARTIALS-1 && !myState->volumeEnvelope.isActive()) {
			// signal no more samples
			unsigned bufferEndIdx = NUM_CH * ((baseIdx + (BUFFER_BLOCK_SIZE - 1)) % CIRCULAR_BUFFER_LEN);
			voiceState->sampleBuffer[bufferEndIdx] = NAN;
		}
	}

	__global__ void evaluateSynthVoiceBlockKernel(SynthState *synthState, unsigned voiceNum, unsigned baseIdx, float fundamentalFreq, bool released) {
		int partialNum = threadIdx.x;
		computePartialOutput(synthState, voiceNum, baseIdx, partialNum, fundamentalFreq, released);
	}

	__host__ void evaluateSynthVoiceBlockOnCpu(float bufferB[BUFFER_BLOCK_SIZE*NUM_CH], unsigned voiceNum, unsigned sampleIdx, float fundamentalFreq, bool released) {
		// need to obtain a lock on the synth state
		std::unique_lock<std::mutex> stateLock(synthStateMutex);
		// move pointer to d_synthState into a local for easy debugging
		SynthState *synthState = d_synthState;
		for (int partialIdx = 0; partialIdx < NUM_PARTIALS; ++partialIdx) {
			computePartialOutput(synthState, voiceNum, sampleIdx, partialIdx, fundamentalFreq, released);
		}
		unsigned bufferStartIdx = NUM_CH * (sampleIdx % CIRCULAR_BUFFER_LEN);
		memcpy(bufferB, &synthState->voiceStates[voiceNum].sampleBuffer[bufferStartIdx], BUFFER_BLOCK_SIZE*NUM_CH*sizeof(float));
	}

	__host__ void evaluateSynthVoiceBlockCuda(float bufferB[BUFFER_BLOCK_SIZE*NUM_CH], unsigned voiceNum, unsigned sampleIdx, float fundamentalFreq, bool released) {
		evaluateSynthVoiceBlockKernel << <1, NUM_PARTIALS >> >(d_synthState, voiceNum, sampleIdx, fundamentalFreq, released);

		checkCudaError(cudaGetLastError()); //check if error in kernel launch
		checkCudaError(cudaDeviceSynchronize()); //check for error INSIDE the kernel

		//copy memory into the cpu buffer
		//Note: this will wait for the kernel to complete first.
		unsigned bufferStartIdx = NUM_CH * (sampleIdx % CIRCULAR_BUFFER_LEN);
		checkCudaError(cudaMemcpy(bufferB, &d_synthState->voiceStates[voiceNum].sampleBuffer[bufferStartIdx], BUFFER_BLOCK_SIZE*NUM_CH*sizeof(float), cudaMemcpyDeviceToHost));
	}

	void evaluateSynthVoiceBlock(float *bufferB, unsigned voiceNum, unsigned baseIdx, float fundamentalFreq, bool released) {
		doStartupOnce();
		if (hasCudaDevice()) {
			evaluateSynthVoiceBlockCuda(bufferB, voiceNum, baseIdx, fundamentalFreq, released);
		} else {
			evaluateSynthVoiceBlockOnCpu(bufferB, voiceNum, baseIdx, fundamentalFreq, released);
		}
	}

	static void memcpyHostToSynthState(void *dest, const void *src, std::size_t numBytes) {
		// if running on device, copy params to GPU via cudaMemcpy, else normal memcpy on cpu.
		if (hasCudaDevice()) {
			// cudaMemcpy is synchronous, so concurrency is dealt with automatically
			checkCudaError(cudaMemcpy(dest, src, numBytes, cudaMemcpyHostToDevice));
		} else {
			// else, copy them using normal memcpy
			// Must first obtain a lock to the synth data.
			std::unique_lock<std::mutex> stateLock(synthStateMutex);
			memcpy(dest, src, numBytes);
		}
	}

	static void memsetSynthState(void *dest, int value, std::size_t numBytes) {
		// if running on device, use cudaMemset, else normal memset
		if (hasCudaDevice()) {
			// cudaMemset is synchronous, so concurrency is dealt with automatically
			checkCudaError(cudaMemset(dest, value, numBytes));
		} else {
			// else, use normal memset
			// Must first obtain a lock to the synth data.
			std::unique_lock<std::mutex> stateLock(synthStateMutex);
			memset(dest, value, numBytes);
		}
	}

	static void copyParameterStates(const ParameterStates *newParameters, ParameterStates *dest) {
		memcpyHostToSynthState(dest, newParameters, sizeof(ParameterStates));
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
		memsetSynthState(&d_synthState->voiceStates[voiceNum].sampleBuffer, 0, CIRCULAR_BUFFER_LEN*NUM_CH*sizeof(float));
	}

}