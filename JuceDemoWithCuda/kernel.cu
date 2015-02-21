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
			F newPhase = phase()*n;
			return ComplexT(newPhase)*newMag;
		}
	};

	// Efficient way to compute successive sine values
	/*class Sinusoidal {
		// The partial has a phase function, phase(t).
		// For constant frequency, phase(t) = w*t.
		// We need varied frequency over time whenever the frequency changes.
		// Thus, dp/dt = w0 + (w1-w0)/T*t, where T is the time over which the frequency should be altered.
		// Write as dp/dt = w0 + kt
		// and so each sample, the phase accumulator should be multiplied by e^i(w0+kt)
		// This *can* be done efficiently.
		// First, evaluate e^iw0 at the start of the block, store as dP/dt
		//   Also evaluate e^ik(1) as d^2P/dt^2. Each sample, multiply dP/dt = dP/dt * d^2P/dt^2
		
		// Use complex float pairs to represent the phase functions
		typedef ComplexT<float> PhaseT;
		PhaseT phase;
		PhaseT phasePrime;
		PhaseT phaseDoublePrime;
	public:
		Sinusoidal() : phase(1, 0) {}
		// transition from start frequency/depth to end frequency/depth over this block
		__device__ __host__ void newFrequencyAndDepth(float startFreq, float endFreq, float startDepth, float endDepth) {
			float wStart = INV_SAMPLE_RATE * startFreq;
			float wEnd = INV_SAMPLE_RATE * endFreq;
			PhaseT phasePrimeStart = PhaseT(wStart);
			PhaseT phasePrimeEnd = PhaseT(wEnd);
			phasePrime = phasePrimeStart;
			// phasePrimeStart * doublePrime^BUFFER_BLOCK_SIZE = phasePrimeEnd
			// (phasePrimeEnd/phasePrimeStart)^(1.0/BUFFER_BLOCK_SIZE) = doublePrime
			// phaseDoublePrime = PhaseT(powf(phasePrimeEnd/phasePrimeStart, 1.0 / BUFFER_BLOCK_SIZE));
			// Note: (a+bi)^n = (r*e^(i*p))^n = r^n*e^(i*n*p)
			phaseDoublePrime = (phasePrimeEnd / phasePrimeStart).pow(1.f / BUFFER_BLOCK_SIZE);

			// we must avoid the division by zero if the current depth is 0.
			// to avoid this, we just prevent the desired depth from ever *being* zero. 
			// In this way, we also don't lose track of position when the depth is toggled to 0
			//   at the cost of some small inaccuracies under specific conditions
			startDepth = max(0.0001, startDepth);
			endDepth = max(0.0001, endDepth);
			// We cannot transition from startDepth to endDepth linearly,
			//   instead we multiply phase by some scalar each frame.
			// so, mag(frame) = startDepth * k^frame
			// and mag(b=BUFFER_BLOCK_SIZE) = endDepth
			// endDepth = startDepth *k^b
			// (endDepth/startDepth)^(1/b) = k
			float k = powf(endDepth / startDepth, 1.f / BUFFER_BLOCK_SIZE);
			// make current phase magnitude equal to startDepth
			float mag = phase.mag();
			float factor = startDepth / mag;
			phase *= factor;
			phasePrime *= k;
		}
		__device__ __host__ PhaseT next() {
			phasePrime *= phaseDoublePrime;
			phase *= phasePrime;
			return phase;
		}
	};*/

	class Sinusoidal {
		// y(t) = mag(t)*sin(phase(t)), all t in frame offset from block start
		// magnitude of sinusoid
		// mag(t) = mag_c0 + t*mag_c1
		float mag_c0;
		float mag_c1;
		// phase function coefficients:
		// phase(t) = phase_c0 + phase_c1*t + phase_c2*t^2
		float phase_c0, phase_c1, phase_c2;
	public:
		Sinusoidal() : mag_c0(0), mag_c1(0), phase_c0(0), phase_c1(0), phase_c2(0) {}
		// startFreq, endFreq given in rad/sec
		__device__ __host__ void newFrequencyAndDepth(float startFreq, float endFreq, float startDepth, float endDepth) {
			// compute phase function coefficients
			// first, carry over the phase from the end of the previous buffer.
			phase_c0 = phaseAtIdx(BUFFER_BLOCK_SIZE);
			// initial slope is w0
			phase_c1 = startFreq*INV_SAMPLE_RATE;
			float endW = endFreq*INV_SAMPLE_RATE;
			// phase'(BUFFER_BLOCK_SIZE) = endW
			// phase_c1 + 2*t*phase_c2 = endW
			// phase_c2 = (endW - phase_c1) / (2*BUFFER_BLOCK_SIZE)
			phase_c2 = (endW - phase_c1) / (2 * BUFFER_BLOCK_SIZE);
			// compute magnitude function coefficients
			mag_c0 = startDepth;
			float deltaDepth = endDepth - startDepth;
			mag_c1 = deltaDepth * INV_BUFFER_BLOCK_SIZE;
			
		}
		__device__ __host__ float phaseAtIdx(unsigned idx) const {
			return phase_c0 + idx*(phase_c1 + idx*phase_c2);
		}
		__device__ __host__ float magAtIdx(unsigned idx) const {
			return mag_c0 + idx*mag_c1;
		}
		__device__ __host__ float valueAtIdx(unsigned idx) const {
			return magAtIdx(idx)*sinf(phaseAtIdx(idx));
		}
	};

	class ADSRState {
		// mode at start of block
		ADSR::Mode mode;
		// break each mode into a line
		// during the attack/decay mode, 
		//   the block may be up to 2 lines during the block. During sustain/release, just one line.
		// actually, for sufficiently short attack/decay, the note may transition from attack->decay->sustain in one single block
		// The best way to handle this is to define the function like:
		// value(t) = (t <= toggleTime)*(line0_c0+line0_c1*t) + !(t <= toggleTime)*(line1_c0+line1_c1*t)
		// toggleIdx must be signed for an easier release mode implementation
		int toggleIdx;
		// segment coefficients
		float line0_c0, line0_c1, line1_c0, line1_c1;
		__device__ __host__ ADSR::Mode nextMode(ADSR::Mode m) const {
			return (ADSR::Mode)((unsigned)m + 1);
		}
	public:
		// initialized at the start of a note
		ADSRState() : mode(ADSR::AttackMode), toggleIdx(BUFFER_BLOCK_SIZE), line0_c0(0), line0_c1(0), line1_c0(0), line1_c1(0) {}
		__device__ __host__ void atBlockStart(ADSR *start, ADSR *end, unsigned partialIdx, bool released) {
			// get the last value from the previous buffer & increment the mode if needed.
			line0_c0 = valueAtIdx(BUFFER_BLOCK_SIZE);
			// increment the mode if we passed the toggleIdx last buffer
			mode = (ADSR::Mode)((unsigned)mode + ((unsigned)mode < (unsigned)ADSR::EndMode && segmentFromIdx(BUFFER_BLOCK_SIZE) == 1));
			// shift toggleIdx - necessary if in release mode, else has no effect.
			toggleIdx -= BUFFER_BLOCK_SIZE;
			if (released && (unsigned)mode < (unsigned)ADSR::ReleaseMode) {
				// change to release mode if the note was released and is in either the attack, decay or sustain mode
				// release must be handled explicitly because the slope depends on the current value obtained.
				mode = ADSR::ReleaseMode;
				float releaseTime = start->getSegmentLength(ADSR::ReleaseMode, partialIdx);
				float endLevel = start->getSegmentStartLevel(ADSR::EndMode);
				float deltaY = endLevel - line0_c0;
				line0_c1 = deltaY / (releaseTime*SAMPLE_RATE);
				line1_c0 = endLevel;
				line1_c1 = 0;
				toggleIdx = releaseTime*SAMPLE_RATE;
			} else if (mode != ADSR::ReleaseMode) {
				// TODO: segments are not linear; they are quadratic
				float startSlope0 = start->getSegmentSlope(mode, partialIdx);
				float startTrigLevel = start->getSegmentStartLevel(nextMode(mode));
				line0_c1 = startSlope0*INV_SAMPLE_RATE;
				float startSlope1 = start->getSegmentSlope(nextMode(mode), partialIdx);
				line1_c0 = startTrigLevel;
				line1_c1 = startSlope1*INV_SAMPLE_RATE;
				// line0(toggleIdx) == line1(toggleIdx)
				// line0_c0 + line0_c1*toggleIdx = line1_c0 + line1_c1*(toggleIdx-toggleIdx);
				// line0_c1*toggleIdx = line1_c0 - line0_c0
				toggleIdx = (line1_c0 - line0_c0) / line0_c1;
			}
		}
		__device__ __host__ bool segmentFromIdx(unsigned idx) const {
			return idx > toggleIdx;
		}
		__device__ __host__ bool isActiveAtEndOfBlock() const {
			return ((unsigned)mode + segmentFromIdx(BUFFER_BLOCK_SIZE)) < (unsigned)ADSR::EndMode;
		}
		__device__ __host__ float valueAtIdx(unsigned idx) const {
			// return the first line evaluated at idx if idx < toggleIdx, else evaluate the second line at idx
			bool seg = segmentFromIdx(idx);
			return (!seg)*(line0_c0 + idx*line0_c1) + (seg)*(line1_c0 + (idx-toggleIdx)*line1_c1);
		}
	};

	class LFOState {
		ADSRState freqAdsrState;
		ADSRState depthAdsrState;
		Sinusoidal sinusoid;
	public:
		__device__ __host__ void atBlockStart(LFO *start, LFO *end, unsigned partialIdx, bool released) {
			ADSR *freqAdsrStart =  start->getFreqAdsr();
			ADSR *depthAdsrStart = start->getDepthAdsr();
			ADSR *freqAdsrEnd =    end->getFreqAdsr();
			ADSR *depthAdsrEnd =   end->getDepthAdsr();
			// update the ADSR states
			freqAdsrState.atBlockStart(freqAdsrStart, freqAdsrEnd, partialIdx, released);
			depthAdsrState.atBlockStart(depthAdsrStart, depthAdsrEnd, partialIdx, released);
			// obtain the starting and ending frequency and depth.
			// We will then just linearly interpolate over the block.
			//float startFreq = freqAdsrState.next();
			//float startDepth = depthAdsrState.next();
			float startFreq = freqAdsrState.valueAtIdx(0);
			float startDepth = depthAdsrState.valueAtIdx(0);
			float endFreq = freqAdsrState.valueAtIdx(BUFFER_BLOCK_SIZE);
			float endDepth = depthAdsrState.valueAtIdx(BUFFER_BLOCK_SIZE);
			sinusoid.newFrequencyAndDepth(startFreq, endFreq, startDepth, endDepth);
			//sinusoid.newFrequencyAndDepth(startFreq, startFreq, startDepth, startDepth);
		}
		__device__ __host__ float valueAtIdx(unsigned idx) const{
			return sinusoid.valueAtIdx(idx);
		}
	};

	class ADSRLFOEnvelopeState {
		ADSRState adsr;
		LFOState lfo;
	public:
		__device__ __host__ void atBlockStart(ADSRLFOEnvelope *envStart, ADSRLFOEnvelope *envEnd, unsigned partialIdx, bool released) {
			adsr.atBlockStart(envStart->getAdsr(), envEnd->getAdsr(), partialIdx, released);
			lfo.atBlockStart(envStart->getLfo(), envEnd->getLfo(), partialIdx, released);
		}
		__device__ __host__ float productAtIdx(unsigned idx) const {
			return adsr.valueAtIdx(idx) * (1 + lfo.valueAtIdx(idx));
		}
		__device__ __host__ float sumAtIdx(unsigned idx) const {
			return adsr.valueAtIdx(idx) + lfo.valueAtIdx(idx);
		}
		__device__ __host__ bool isActiveAtEndOfBlock() const {
			return adsr.isActiveAtEndOfBlock();
		}
	};

	class DetuneEnvelopeState {
		ADSRLFOEnvelopeState adsrLfoState;
	public:
		__device__ __host__ void atBlockStart(DetuneEnvelope *envStart, DetuneEnvelope *envEnd, unsigned partialIdx, bool released) {
			adsrLfoState.atBlockStart(envStart->getAdsrLfo(), envEnd->getAdsrLfo(), partialIdx, released);
		}
		__device__ __host__ float valueAtIdx(unsigned idx) const {
			return adsrLfoState.sumAtIdx(idx);
		}
	};

	// Contains info about the parameter states at ANY sample in the block
	struct FullBlockParameterInfo {
		ParameterStates start;
		ParameterStates end;
	};

	// Contains extra state information relevant to each individual partial
	struct PartialState {
		Sinusoidal sinusoid;
		ADSRLFOEnvelopeState volumeEnvelope;
		ADSRLFOEnvelopeState stereoPanEnvelope;
		DetuneEnvelopeState detuneEnvelope;
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
		ParameterStates *startParams = &voiceState->parameterInfo.start;
		ParameterStates *endParams = &voiceState->parameterInfo.end;
		detuneEnvelope.atBlockStart(&startParams->detuneEnvelope, &endParams->detuneEnvelope, partialIdx, released);
		
		// calculate the start and end frequency for this block
		float baseFreq = (partialIdx + 1)*fundamentalFreq;
		float detuneStart = detuneEnvelope.valueAtIdx(0);
		float detuneEnd = detuneEnvelope.valueAtIdx(BUFFER_BLOCK_SIZE);
		//float detuneStart = 0.f;
		//float detuneEnd = 0.f;

		// configure the sinusoid to transition from the starting frequency to the end frequency
		sinusoid.newFrequencyAndDepth(baseFreq*(1.f+detuneStart), baseFreq*(1.f+detuneEnd), 1.f, 1.f);
		volumeEnvelope.atBlockStart(&startParams->volumeEnvelope, &endParams->volumeEnvelope, partialIdx, released);
		stereoPanEnvelope.atBlockStart(&startParams->stereoPanEnvelope, &endParams->stereoPanEnvelope, partialIdx, released);
	}

	// this is a circular buffer of sample data (interleaved by channel number) stored on the device
	// It is persistent and lengthy, in order to accomodate the delay effect.
	SynthState *d_synthState = NULL;

	// When running on the cpu, we need to control concurrent access to the synth state
	std::mutex synthStateMutex;

	static void printCudaDeviveProperties(cudaDeviceProp devProp) {
		// utility function to log device info. Source: https://www.cac.cornell.edu/vw/gpu/example_submit.aspx
		printf("Major revision number:         %d\n", devProp.major);
		printf("Minor revision number:         %d\n", devProp.minor);
		printf("Name:                          %s\n", devProp.name);
		printf("Total global memory:           %lu\n", devProp.totalGlobalMem);
		printf("Total shared memory per block: %lu\n", devProp.sharedMemPerBlock);
		printf("Total registers per block:     %d\n", devProp.regsPerBlock);
		printf("Warp size:                     %d\n", devProp.warpSize);
		printf("Maximum memory pitch:          %lu\n", devProp.memPitch);
		printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
		for (int i = 0; i < 3; ++i) {
			printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
		}
		for (int i = 0; i < 3; ++i) {
			printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
		}
		printf("Clock rate:                    %d\n", devProp.clockRate);
		printf("Total constant memory:         %lu\n", devProp.totalConstMem);
		printf("Texture alignment:             %lu\n", devProp.textureAlignment);
		printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
		printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
		printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
	}


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
		bool useCuda = (err == cudaSuccess && deviceCount != 0) && !NEVER_USE_CUDA;
		printf("Using Cuda? %i\n", useCuda);
		if (useCuda) {
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, 0);
			printCudaDeviveProperties(prop);
		}
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
		// This reduction method requires a temporary array in shared memory.
		__shared__ float partialReductionOutputs[NUM_PARTIALS*NUM_CH];

		partialReductionOutputs[NUM_CH*partialIdx + 0] = outputL;
		partialReductionOutputs[NUM_CH*partialIdx + 1] = outputR;
		unsigned numActiveThreads = NUM_PARTIALS / 2;
		while (numActiveThreads > 0) {
			__syncthreads();
			if (partialIdx < numActiveThreads) {
				partialReductionOutputs[NUM_CH*partialIdx + 0] += partialReductionOutputs[NUM_CH*partialIdx + numActiveThreads*NUM_CH + 0];
				partialReductionOutputs[NUM_CH*partialIdx + 1] += partialReductionOutputs[NUM_CH*partialIdx + numActiveThreads*NUM_CH + 1];
			}
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
		if (partialIdx == NUM_PARTIALS-1) {
			// TODO: only do this copy if the parameters have changed
			memcpy(&voiceState->parameterInfo.start, &voiceState->parameterInfo.end, sizeof(ParameterStates));
		}

	}

	// compute the output for ONE sine wave over the current block
	__device__ __host__ void computePartialOutput(SynthState *synthState, unsigned voiceNum, unsigned baseIdx, unsigned partialIdx, float fundamentalFreq, bool released) {
		SynthVoiceState *voiceState = &synthState->voiceStates[voiceNum];
		PartialState* myState = &voiceState->partialStates[partialIdx];
		myState->atBlockStart(voiceState, partialIdx, fundamentalFreq, released);
		// Get the base partial level (the hand-drawn frequency weights)
		float level = (1.0 / NUM_PARTIALS) * voiceState->parameterInfo.start.partialLevels[partialIdx];
		//printf("partialIdx: %i\n", partialIdx);
		for (int sampleIdx = 0; sampleIdx < BUFFER_BLOCK_SIZE; ++sampleIdx) {
			// Extract the sinusoidal portion of the wave.
			// float sinusoid = myState->sinusoid.next().imag();
			float sinusoid = myState->sinusoid.valueAtIdx(sampleIdx);
			// Get the ADSR/LFO volume envelope
			//float envelope = myState->volumeEnvelope.nextAsProduct();
			//float pan = myState->stereoPanEnvelope.nextAsSum();
			float envelope = myState->volumeEnvelope.productAtIdx(sampleIdx);
			float pan = myState->stereoPanEnvelope.sumAtIdx(sampleIdx);
			float unpanned = level*envelope*sinusoid;
			//float outputL = unpanned;
			//float outputR = unpanned;
			// full left = -1 pan. full right = +1 pan.
			// Use circular panning, where L^2 + R^2 = 1.0
			//   R(+1.0 pan) = 1.0, L(-1.0 pan) = 0.0, R(0.0 pan) = sqrt(1/2)
			//   L(+1.0 pan) = 0.0, L(-1.0 pan) = 1.0, L(0.0 pan) = sqrt(1/2)
			//   then R(pan) = sqrt((1+pan)/2)
			//   L(pan) = sqrt((1-pan)/2)
			// Note that L(pan)^2 + R(pan)^2 = 1.0, so energy is constant.
			// Must deal with pan values of magnitude > 1.0
			// Note the analog between sinusoidals:
			// sin(x)^2 + cos(x)^2 = 1.0 = L(pan)^2 + R(pan)^2
			// sin(pi/4) = cos(pi/4) = L(0.0) = R(0.0) = sqrt(1/2)
			// cos(0.0) = L(-1.0) = 1.0
			// cos(pi/2) = L(1.0) = 0.0
			// sin(0.0) = R(-1.0) = 0.0
			// sin(pi/2) = R(1.0) = 1.0
			// So, L(pan) = cos(pi/4 + pi/4*pan) = cos(pi/4*(1+pan))
			//     R(pan) = sin(pi/4 + pi/4*pan) = sin(pi/4*(1+pan))
			float angle = PI / 4 * (1 + pan);
			float outputL = unpanned * cosf(angle);
			float outputR = unpanned * sinf(angle);
			//float outputL = unpanned * sqrt(0.5*(1-pan));
			//float outputR = unpanned * sqrt(0.5*(1+pan));
			//linear pan implementation:
			//float outputL = unpanned * 0.5*(1 - pan);
			//float outputR = unpanned * 0.5*(1 + pan);

			reduceOutputs(voiceState, partialIdx, baseIdx + sampleIdx, outputL, outputR);
		}
		updateVoiceParametersIfNeeded(voiceState, voiceNum, partialIdx);
		// TODO: use a proper reduction algorithm to determine when the note is complete
		if (partialIdx == NUM_PARTIALS-1 && !myState->volumeEnvelope.isActiveAtEndOfBlock()) {
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