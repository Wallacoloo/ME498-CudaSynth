#include "kernel.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <string.h> // for memset
#include <assert.h>
#include <stdlib.h> // for atexit
#include <mutex>
#include <thread> // for unique_lock
#include <random> // for deterministic pseudorandom number generation

#include "defines.h"

#define CIRCULAR_BUFFER_LEN MAX_DELAY_EFFECT_LENGTH

namespace kernel {
	// define statics
	unsigned ParameterStates::nextUUID(0);

	// forward-declare necessary classes
	struct SynthVoiceState;
	struct SynthState;
	// this is a circular buffer of sample data (interleaved by channel number) stored on the device
	// It is persistent and lengthy, in order to accomodate the delay effect.
	SynthState *d_synthState = NULL;

	// When running on the cpu, we need to control concurrent access to the synth state
	std::mutex synthStateMutex;

	class Sinusoidal {
		// y(t) = mag(t)*sin(phase(t)), all t in frame offset from block start
		// magnitude of sinusoid
		// mag(t) = mag_c0 + t*mag_c1
		float mag_c0;
		float mag_c1;
		// phase function coefficients:
		// phase(t) = phase_c0 + phase_c1*t + phase_c2*t^2
		float phase_c0, phase_c1, phase_c2;
		__device__ __host__ float phaseAtIdx(unsigned idx) const {
			return phase_c0 + idx*(phase_c1 + idx*phase_c2);
		}
		__device__ __host__ float magAtIdx(unsigned idx) const {
			return mag_c0 + idx*mag_c1;
		}
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
		__device__ __host__ float valueAtIdx(unsigned idx) const {
			return magAtIdx(idx)*sinf(phaseAtIdx(idx));
		}
		__device__ __host__ float freqAtIdx(unsigned idx) const {
			// freq = d/dt (phase)
			return phase_c1 + 2 * phase_c2*idx;
		}
	};

	class RandomNumberGen {
		float randomValues[DETUNE_NUM_SEEDS][NUM_PARTIALS];
	public:
		RandomNumberGen() {
			// want randomValues[i][j] to stay the same regardless of NUM_PARTIALS,
			// so process each row with an independent seed.
			// create the seeds with ANOTHER random number generator
			std::minstd_rand seedGen(119606366); // seed chosen from random.org.
			std::minstd_rand rng;
			for (int row = 0; row < DETUNE_NUM_SEEDS; ++row) {
				rng.seed(seedGen());
				for (int partial = 0; partial < NUM_PARTIALS; ++partial) {
					// generate a normalized random number (0 - 1)
					float normRand = (float)rng() / 2147483647;
					// turn this into a symmetric distribution from (-1, 1) centered at 0.
					float doubleSided = -1 + 2 * normRand;
					randomValues[row][partial] = doubleSided;
				}
			}
		}
		// return a random number from interpolated the N seeds evaluated at partialIdx.
		// seedNo should be between [0, 1]
		__host__ __device__ float getFor(float seedNo, unsigned partialIdx) {
			// Interpolate the seeds using the following algorithm:
			// v(seed, partial) = (1 - seed^2)*seed0[partial] + (1 - (seed-1/N))*seed1[partial] + (1 - (seed-2/N))*seed2[partial] + ...
			// where N is the number of seeds MINUS 1.
			float value = 0.f;
			for (int curSeed = 0; curSeed < DETUNE_NUM_SEEDS; ++curSeed) {
				float distance = seedNo - curSeed / (float)(DETUNE_NUM_SEEDS - 1);
				float weight = 1 - distance*distance;
				value += weight * randomValues[curSeed][partialIdx];
			}
			return value;
		}
	};

	class ADSRState {
		// better approach (not yet implemented):
		//   upon ADSR change:
		//     determine the current segment, current length, current value and current time (current time MUST be stored)
		//     determine the new end value and new length
		//     alter coefficients such that current values match and so that the end value will be reached at the new length.
		// proportion of way through ADSR envelope at start;
		// 0 <= P < 1 for attack phase,
		// 1 <= P < 2 for decay phase,
		// 2 <= P < 3 for sustain phase,
		// 3 <= P < 4 for release phase,
		// 4 <= P indicates note end.
		float P;
		// break each mode into a line
		// during the attack/decay mode, 
		//   the block may be up to 2 lines during the block. During sustain/release, just one line.
		// actually, for sufficiently short attack/decay, the note may transition from attack->decay->sustain in one single block
		// in this case, it is sufficient to clamp the index to the point at which we switch from decay to sustain mode, since decay(last) == sustain
		// The best way to handle this is to define the function like:
		// value(t) = (t <= toggleTime)*(line0_c0+line0_c1*t+line0_c2*t^2) + !(t <= toggleTime)*(line1_c0+line1_c1*t+line1_c2*t^2)
		// any index > clampIdx should return the same value as clampIdx. This is for handling 3-part envelopes where the final portion is constant.
		float clampP;
		// segment coefficients
		// the values these take are in the same units as 'P'
		float line0_c0, line0_c1, line1_c0, line1_c1;
		float line0_invLength;
		float line1_invLength;
		__device__ __host__ ADSR::Mode getMode() const {
			return (ADSR::Mode)(unsigned)P;
		}
		__device__ __host__ ADSR::Mode nextMode(ADSR::Mode m) const {
			return (ADSR::Mode)((unsigned)m + 1);
		}
		__device__ __host__ float unclampedPFromIdx(float idx) const {
			return P + idx*line0_invLength;
		}
		__device__ __host__ float pFromIdx(float idx) const {
			float pIdx = unclampedPFromIdx(idx);
			return min(pIdx, clampP);
		}
		__device__ __host__ bool segmentFromP(float pIdx) const {
			return pIdx >= (unsigned)nextMode(getMode());
		}
		__device__ __host__ float interpolate(float position, float a, float b) const {
			// construct a function where f(0) = a, f(1) = b, and return f(position)
			return a + (b - a)*position;
		}
	public:
		// initialized at the start of a note
		ADSRState() : P(0), clampP(2.f), 
			line0_c0(0), line0_c1(0), 
			line1_c0(0), line1_c1(0),
			line0_invLength(1e-7f), line1_invLength(1e-7f) {}
		__device__ __host__ void atBlockStart(ADSR *start, ADSR *end, unsigned partialIdx, bool released, bool didParamsChange) {
			// preserve previous value
			float prevValue = valueAtIdx(BUFFER_BLOCK_SIZE);
			// track position in envelope
			float idxOfSwitch = ((unsigned)nextMode(getMode()) - P) / line0_invLength;
			idxOfSwitch = min(idxOfSwitch, (float)BUFFER_BLOCK_SIZE);
			// add accumulated index change from each segment
			P = min(clampP, P + idxOfSwitch*line0_invLength + (BUFFER_BLOCK_SIZE - idxOfSwitch)*line1_invLength);
			// if we're released, skip to release mode (or further)
			P = max(P, released*(float)(unsigned)ADSR::ReleaseMode);
			// update slope of segment and rate at which we progress:
			float line0_length = end->getSegmentLength(getMode(), partialIdx) * SAMPLE_RATE;
			line0_invLength = 1.f / line0_length;
			float line1_length = end->getSegmentLength(nextMode(getMode()), partialIdx) * SAMPLE_RATE;
			line1_invLength = 1.f / line1_length;
			// calculate endpoint values for our lines
			float line0_endPointX, line0_endPointY;
			// float line0_relPositionAtBufferBlockSize = pFromIdx(BUFFER_BLOCK_SIZE) - (float)(unsigned)getMode();
			// float line0_valueAtBufferBlockSize = interpolate(line0_relPositionAtBufferBlockSize, end->getSegmentStartLevel(getMode()), end->getSegmentStartLevel(nextMode(getMode())));
			if ((unsigned)P == (unsigned)ADSR::SustainMode || (unsigned)P == (unsigned)ADSR::EndMode) {
				line0_endPointX = unclampedPFromIdx(BUFFER_BLOCK_SIZE);
				line0_endPointY = interpolate(line0_endPointX - (float)(unsigned)getMode(), end->getSegmentStartLevel(getMode(), partialIdx), end->getSegmentStartLevel(nextMode(getMode()), partialIdx));
			} else {
				line0_endPointX = (float)(unsigned)nextMode(getMode());
				line0_endPointY = end->getSegmentStartLevel(nextMode(getMode()), partialIdx);
			}
			float line1_startValue = end->getSegmentStartLevel(nextMode(getMode()), partialIdx);
			// update c0 and c1 based on the following constraints:
			// value(P) == prevValue
			// value(endPointX) == endPointY
			// c0 + c1*P == prevValue
			// c0 + c1*P2 == endValue
			// c1*(P2-P) == endValue-prevValue -> c1 = (endValue-prevValue)/(P2-P)
			// c0 = prevValue - c1*P;
			line0_c1 = (line0_endPointY - prevValue) / (line0_endPointX-P);
			line0_c0 = prevValue - line0_c1*P;
			// then calculate the coefficients for the second portion of the line
			// line1(endP) == startVal
			// line1(endP+length1*sample_rate*IL0) == endVal
			unsigned endP = (unsigned)nextMode(getMode());
			float line1_endValue = end->getSegmentStartLevel(nextMode(nextMode(getMode())), partialIdx);
			// c0 + c1*endP == startVal
			// c0 + c1*endP + c1*length1*IL0 == endVal
			// c1*length1*IL0 == endVal - startVal
			line1_c1 = (line1_endValue - line1_startValue) / (line1_length*line0_invLength);
			// line1_c0 + line1_c1*endP == startValue
			line1_c0 = line1_startValue - line1_c1*endP;
			// then determine the value for clampP
			// endP+length1*IL0 == P+clampIdx*IL0
			// (endP-P)/IL0 + length1 = clampIdx
			// endP+length1*IL0 == clampP
			//float seg1StartIdx = (endP - P) / line0_invLength;
			//float seg1EndIdx = seg1StartIdx + line1_length;
			//clampIdx = seg1EndIdx;
			clampP = endP + line1_length*line0_invLength;
		}
		__device__ __host__ bool isActiveAtEndOfBlock() const {
			return pFromIdx(BUFFER_BLOCK_SIZE) < (unsigned)ADSR::EndMode;
		}
		__device__ __host__ float valueAtIdx(unsigned idx) const {
			// return either the first or second line evaluated at idx, depending on where the switch occurs
			float pIdx = pFromIdx((float)idx);
			bool seg = segmentFromP(pIdx);
			return (!seg)*(line0_c0 + pIdx*line0_c1) + (seg)*(line1_c0 + pIdx*line1_c1);
		}
	};

	class LFOState {
		ADSRState freqAdsrState;
		ADSRState depthAdsrState;
		Sinusoidal sinusoid;
	public:
		__device__ __host__ void atBlockStart(LFO *start, LFO *end, unsigned partialIdx, bool released, bool didParamsChange) {
			ADSR *freqAdsrStart =  start->getFreqAdsr();
			ADSR *depthAdsrStart = start->getDepthAdsr();
			ADSR *freqAdsrEnd =    end->getFreqAdsr();
			ADSR *depthAdsrEnd =   end->getDepthAdsr();
			// update the ADSR states
			freqAdsrState.atBlockStart(freqAdsrStart, freqAdsrEnd, partialIdx, released, didParamsChange);
			depthAdsrState.atBlockStart(depthAdsrStart, depthAdsrEnd, partialIdx, released, didParamsChange);
			// obtain the starting and ending frequency and depth.
			// We will then just linearly interpolate over the block.
			float startFreq = freqAdsrState.valueAtIdx(0);
			float startDepth = depthAdsrState.valueAtIdx(0);
			float endFreq = freqAdsrState.valueAtIdx(BUFFER_BLOCK_SIZE);
			float endDepth = depthAdsrState.valueAtIdx(BUFFER_BLOCK_SIZE);
			sinusoid.newFrequencyAndDepth(startFreq, endFreq, startDepth, endDepth);
		}
		__device__ __host__ float valueAtIdx(unsigned idx) const{
			return sinusoid.valueAtIdx(idx);
		}
	};

	class ADSRLFOEnvelopeState {
		ADSRState adsr;
		LFOState lfo;
	public:
		__device__ __host__ void atBlockStart(ADSRLFOEnvelope *envStart, ADSRLFOEnvelope *envEnd, unsigned partialIdx, bool released, bool didParamsChange) {
			adsr.atBlockStart(envStart->getAdsr(), envEnd->getAdsr(), partialIdx, released, didParamsChange);
			lfo.atBlockStart(envStart->getLfo(), envEnd->getLfo(), partialIdx, released, didParamsChange);
		}
		__device__ __host__ float adsrAtIdx(unsigned idx) const {
			return adsr.valueAtIdx(idx);
		}
		__device__ __host__ float lfoAtIdx(unsigned idx) const {
			return lfo.valueAtIdx(idx);
		}
		__device__ __host__ float productAtIdx(unsigned idx) const {
			return adsrAtIdx(idx) * (1 + lfoAtIdx(idx));
		}
		__device__ __host__ float sumAtIdx(unsigned idx) const {
			return adsrAtIdx(idx) + lfoAtIdx(idx);
		}
		__device__ __host__ bool isActiveAtEndOfBlock() const {
			return adsr.isActiveAtEndOfBlock();
		}
	};

	class DetuneEnvelopeState {
		ADSRLFOEnvelopeState adsrLfoState;
		float weight;
	public:
		__device__ __host__ void atBlockStart(SynthState *synthState, DetuneEnvelope *envStart, DetuneEnvelope *envEnd, unsigned partialIdx, bool released, bool didParamsChange);
		__device__ __host__ float valueAtIdx(unsigned idx) const {
			return weight*adsrLfoState.sumAtIdx(idx);
		}
	};

	class DelayEnvelopeState {
		ADSRLFOEnvelopeState spaceBetweenEchoes;
		ADSRLFOEnvelopeState amplitudeLostPerEcho;
	public:
		__device__ __host__ void atBlockStart(DelayEnvelope *envStart, DelayEnvelope *envEnd, unsigned partialIdx, bool released, bool didParamsChange) {
			spaceBetweenEchoes.atBlockStart(envStart->getSpaceBetweenEchoes(), envEnd->getSpaceBetweenEchoes(), partialIdx, released, didParamsChange);
			amplitudeLostPerEcho.atBlockStart(envStart->getAmplitudeLostPerEcho(), envEnd->getAmplitudeLostPerEcho(), partialIdx, released, didParamsChange);
		}
		__device__ __host__ float spaceBetweenEchoesAtIdx(unsigned idx) const {
			//return spaceBetweenEchoes.adsrAtIdx(idx);
			return spaceBetweenEchoes.productAtIdx(idx);
		}
		__device__ __host__ float amplitudeLostPerEchoAtIdx(unsigned idx) const {
			//return amplitudeLostPerEcho.adsrAtIdx(idx);
			return amplitudeLostPerEcho.productAtIdx(idx);
		}
	};

	class FilterState {
		ADSRState shiftState;
		// have a bunch of piecewise linear functions.
		// can split into y(w) = sum of yn(w)
		// where yn(w) = { an*w + bn, Ln < w < Rn
		//			       0, otherwise }
		// This is achieved via two comparisons and two multiplies on top of evaluating an*w + bn.
		// Alternatively:
		// yn(w) = an*clamp(w, Ln, Rn) + bn
		// This is just 2 extra min/max calls (same cost as a comparison)
		// The catch is that yn(w) is not zero outside of its active domain
		// we can actually go further:
		// yn(w) = an*max(w, Ln) + bn
		// merge the constants:
		// y(w) = sum[an*max(w, Ln)] + b
		// determining the coefficients becomes slightly more difficult. 
		// an can be solved by knowing the slope along each interval.
		// b can be solved by substituting y(0) = sum[an*Ln] + b
		struct Piece {
			float beginTime;
			float slope;
		};
		Piece pieces[PIECEWISE_MAX_PIECES];
		float b;
		float freq_c0, freq_c1;
	public:
		__device__ __host__ void atBlockStart(FilterEnvelope *envStart, FilterEnvelope *envEnd, float freqStart, float freqEnd, bool released, bool didParamsChange) {
			shiftState.atBlockStart(envStart->getShift(), envEnd->getShift(), 0, released, didParamsChange);
			// set the frequency coefficients such that:
			// w(idx) = freq_c0 + freq_c1*idx
			// w(0) = freqStart,
			// w(BUFFER_BLOCK_SIZE) = freqEnd,
			freq_c0 = freqStart;
			freq_c1 = (freqEnd - freqStart) * INV_BUFFER_BLOCK_SIZE;
			// determine the coefficients.
			// no filter interpolation for now, since that requires doubling the number of nodes
			PiecewiseFunction *func = envEnd->getShape();
			unsigned numActivePieces = func->numPoints();
			float y0 = func->startLevelOfPiece(0);
			float offsetSum = 0;
			float prevSlope = 0.f;
			for (unsigned i = 0; i < numActivePieces; ++i) {
				float thisBeginTime = func->startTimeOfPiece(i);
				float nextTime = func->startTimeOfPiece(i + 1);
				float thisLength = nextTime - thisBeginTime;
				float overallSlope = (i + 1 == numActivePieces) ? 0.f
					: (func->startLevelOfPiece(i + 1) - func->startLevelOfPiece(i)) / thisLength;
				float thisSlope = overallSlope - prevSlope;
				prevSlope = overallSlope;
				pieces[i].slope = thisSlope;
				pieces[i].beginTime = thisBeginTime;
				offsetSum += thisSlope*thisBeginTime;
			}
			// determine the coefficient 'b':
			// y(0) = sum[slope_n*beginTime_n] + b
			// b = y(0) - sum[slope_n*beginTime_n]
			b = y0 - offsetSum;
			// zero contributions from inactive pieces
			for (unsigned i = numActivePieces; i < PIECEWISE_MAX_PIECES; ++i) {
				pieces[i].slope = 0;
				pieces[i].beginTime = 0;
			}
		}
		__device__ __host__ float valueAtIdx(unsigned idx) const {
			// y(w) = sum[an*max(w, Ln)] + b
			float sum = b;
			float w = freq_c0 + idx*freq_c1;
			// transpose the envelope by shiftin the frequency
			w -= shiftState.valueAtIdx(idx);
			for (int i = 0; i < PIECEWISE_MAX_PIECES; ++i) {
				sum += pieces[i].slope * max(w, pieces[i].beginTime);
			}
			return sum;
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
		DelayEnvelopeState delayState;
		FilterState filterState;
		PartialState() {}
		PartialState(struct SynthState *synthState, unsigned voiceNum, unsigned partialIdx) {}
		__device__ __host__ void atBlockStart(SynthState *synthState, SynthVoiceState *voiceState, unsigned partialIdx, float fundamentalFreq, bool released);
	};

	struct SynthVoiceState {
		FullBlockParameterInfo parameterInfo;
		float sampleBuffer[CIRCULAR_BUFFER_LEN*NUM_CH];
		// assume the GPU will require more threads than CPU,
		// so allocate enough space for either CPU or GPU implementation
		PartialState partialStates[NUM_THREADS_PER_PARTIAL_GPU][NUM_PARTIALS];
		SynthVoiceState() {
			memset(sampleBuffer, 0, sizeof(sampleBuffer));
		}
	};

	// Packages all the state-related information for the synth in one class to store persistently on the device
	struct SynthState {
		RandomNumberGen randomNumbers;
		SynthVoiceState voiceStates[MAX_SIMULTANEOUS_SYNTH_NOTES];
	};

	__host__ __device__ void DetuneEnvelopeState::atBlockStart(SynthState *synthState, DetuneEnvelope *envStart, DetuneEnvelope *envEnd, unsigned partialIdx, bool released, bool didParamsChange) {
		float randDepth = envStart->getRandMix();
		float randOffset = synthState->randomNumbers.getFor(envStart->getRandSeed(), partialIdx);
		weight = 1 + (randOffset - 1)*randDepth;
		adsrLfoState.atBlockStart(envStart->getAdsrLfo(), envEnd->getAdsrLfo(), partialIdx, released, didParamsChange);
	}

	void PartialState::atBlockStart(SynthState *synthState, SynthVoiceState *voiceState, unsigned partialIdx, float fundamentalFreq, bool released) {
		ParameterStates *startParams = &voiceState->parameterInfo.start;
		ParameterStates *endParams = &voiceState->parameterInfo.end;
		bool didParamsChange = (voiceState->parameterInfo.start.UUID != voiceState->parameterInfo.end.UUID);

		// init detune envelope
		detuneEnvelope.atBlockStart(synthState, &startParams->detuneEnvelope, &endParams->detuneEnvelope, partialIdx, released, didParamsChange);
		
		// init delay state
		delayState.atBlockStart(&startParams->delayEnvelope, &endParams->delayEnvelope, partialIdx, released, didParamsChange);

		// calculate the start and end frequency for this block
		float baseFreq = (partialIdx + 1)*fundamentalFreq;
		float detuneStart = detuneEnvelope.valueAtIdx(0);
		float detuneEnd = detuneEnvelope.valueAtIdx(BUFFER_BLOCK_SIZE);
		float freqStart = baseFreq*(1.f + detuneStart);
		float freqEnd = baseFreq*(1.f + detuneEnd);

		// configure the sinusoid to transition from the starting frequency to the end frequency
		sinusoid.newFrequencyAndDepth(freqStart, freqEnd, 1.f, 1.f);
		volumeEnvelope.atBlockStart(&startParams->volumeEnvelope, &endParams->volumeEnvelope, partialIdx, released, didParamsChange);
		stereoPanEnvelope.atBlockStart(&startParams->stereoPanEnvelope, &endParams->stereoPanEnvelope, partialIdx, released, didParamsChange);
		filterState.atBlockStart(&startParams->filterEnvelope, &endParams->filterEnvelope, freqStart, freqEnd, released, didParamsChange);
	}

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

	int numThreadsPerPartial() {
		return hasCudaDevice() ? NUM_THREADS_PER_PARTIAL_GPU : NUM_THREADS_PER_PARTIAL_CPU;
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
		std::unique_lock<std::mutex> stateLock(synthStateMutex);
		SynthState *defaultState = new SynthState();
		if (hasCudaDevice()) {
			// allocate sample buffer on device
			checkCudaError(cudaMalloc(&d_synthState, sizeof(SynthState)));
			checkCudaError(cudaMemcpy(d_synthState, defaultState, sizeof(SynthState), cudaMemcpyHostToDevice));
		} else {
			// allocate sample buffer on cpu
			d_synthState = (SynthState*)malloc(sizeof(SynthState));
			memcpy(d_synthState, defaultState, sizeof(SynthState));
		}
		delete defaultState;
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
			// zero the previous frame's outputs so delay effect can fill them
			unsigned prevIdx = NUM_CH * ((CIRCULAR_BUFFER_LEN + sampleIdx - BUFFER_BLOCK_SIZE) % CIRCULAR_BUFFER_LEN);
			
			voiceState->sampleBuffer[prevIdx + 0] = 0;
			voiceState->sampleBuffer[prevIdx + 1] = 0;
			//atomicExch(&voiceState->sampleBuffer[prevIdx + 0], 0);
			//atomicExch(&voiceState->sampleBuffer[prevIdx + 1], 0);
			// add output to buffer (atomically)
			atomicAdd(&voiceState->sampleBuffer[bufferIdx + 0], partialReductionOutputs[0]);
			atomicAdd(&voiceState->sampleBuffer[bufferIdx + 1], partialReductionOutputs[1]);
			//unsigned nextIdx = NUM_CH * ((sampleIdx + 40000) % (CIRCULAR_BUFFER_LEN));
			//atomicAdd(&voiceState->sampleBuffer[nextIdx + 0], partialReductionOutputs[0]);
			//atomicAdd(&voiceState->sampleBuffer[nextIdx + 1], partialReductionOutputs[1]);
		}
#else
		//host code
		//Since everything's computed iteratively, we can just add our outputs directly to the buffer.
		//First write to this sample must zero-initialize the buffer (not required in the GPU code).
		if (partialIdx == 0) {
			// zero the previous frame's outputs so delay effect can fill them
			unsigned prevIdx = NUM_CH * ((CIRCULAR_BUFFER_LEN + sampleIdx - BUFFER_BLOCK_SIZE) % CIRCULAR_BUFFER_LEN);
			voiceState->sampleBuffer[prevIdx + 0] = 0;
			voiceState->sampleBuffer[prevIdx + 1] = 0;
		}
		voiceState->sampleBuffer[bufferIdx + 0] += outputL;
		voiceState->sampleBuffer[bufferIdx + 1] += outputR;
#endif
	}

	// called for each partial to sum their outputs together.
	__device__ __host__ void reduceDelayOutputs(SynthVoiceState *voiceState, unsigned partialIdx, int sampleIdx, float outputL, float outputR) {
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
		/*__shared__ float partialReductionOutputs[NUM_PARTIALS*NUM_CH];

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
			// add output to buffer (atomically)
			atomicAdd(&voiceState->sampleBuffer[bufferIdx + 0], partialReductionOutputs[0]);
			atomicAdd(&voiceState->sampleBuffer[bufferIdx + 1], partialReductionOutputs[1]);
			//unsigned nextIdx = NUM_CH * ((sampleIdx + 40000) % (CIRCULAR_BUFFER_LEN));
			//atomicAdd(&voiceState->sampleBuffer[nextIdx + 0], partialReductionOutputs[0]);
			//atomicAdd(&voiceState->sampleBuffer[nextIdx + 1], partialReductionOutputs[1]);
		}*/
		atomicAdd(&voiceState->sampleBuffer[bufferIdx + 0], outputL);
		atomicAdd(&voiceState->sampleBuffer[bufferIdx + 1], outputR);
#else
		//host code
		//Since everything's computed iteratively, we can just add our outputs directly to the buffer.
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
		if (voiceState->parameterInfo.start.UUID != voiceState->parameterInfo.end.UUID) {
			if (partialIdx == NUM_PARTIALS - 1) {
				memcpy(&voiceState->parameterInfo.start, &voiceState->parameterInfo.end, sizeof(ParameterStates));
			}
		}
	}

	__device__ __host__ float antiAliasedVolumeForFreq(float angularFreq) {
		float falloffWidth = 4000.f;
		float invFalloffWidth = 0.00025f;
		float falloffEnd = 0.5f*SAMPLE_RATE_RAD;
		float falloffStart = falloffEnd - falloffWidth;
		
		float clamped = min(falloffEnd, max(falloffStart, angularFreq));
		float level = 1.f - (clamped - falloffStart) * invFalloffWidth;
		return level;
	}

	// compute the output for ONE sine wave over the current block
	__device__ __host__ void computePartialOutput(SynthState *synthState, unsigned voiceNum, unsigned baseIdx, unsigned partialIdx, unsigned samplesPerThread, unsigned threadIdWithinPartial, float fundamentalFreq, bool released) {
		SynthVoiceState *voiceState = &synthState->voiceStates[voiceNum];
		PartialState* myState = &voiceState->partialStates[threadIdWithinPartial][partialIdx];
		myState->atBlockStart(synthState, voiceState, partialIdx, fundamentalFreq, released);
		// Get the base partial level (the hand-drawn frequency weights)
		float level = (1.f / NUM_PARTIALS) * voiceState->parameterInfo.start.partialLevels[partialIdx];
		//printf("partialIdx: %i\n", partialIdx);
		for (unsigned sampleIdx = threadIdWithinPartial*samplesPerThread; sampleIdx < (threadIdWithinPartial+1)*samplesPerThread; ++sampleIdx) {
			// Extract the sinusoidal portion of the wave.
			float sinusoid = myState->sinusoid.valueAtIdx(sampleIdx);
			// Compute a secondary envelope that prevents aliasing
			float freq = myState->sinusoid.freqAtIdx(sampleIdx);
			float filterEnv = myState->filterState.valueAtIdx(sampleIdx);
			float antiAliasEnv = antiAliasedVolumeForFreq(freq);
			// Get the ADSR/LFO volume envelope
			float envelope = antiAliasEnv*filterEnv*myState->volumeEnvelope.productAtIdx(sampleIdx);
			float pan = myState->stereoPanEnvelope.sumAtIdx(sampleIdx);
			float unpanned = level*envelope*sinusoid;
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
			float angle = PIf / 4 * (1 + pan);
			float outputL = unpanned * cosf(angle);
			float outputR = unpanned * sinf(angle);
			// alternative linear pan implementation:
			// float outputL = unpanned * 0.5*(1 - pan);
			// float outputR = unpanned * 0.5*(1 + pan);
			// write the output to the buffer, using a reduction algorithm to avoid serialization
			reduceOutputs(voiceState, partialIdx, baseIdx + sampleIdx, outputL, outputR);
			// compute delays
			float delayPerEcho = myState->delayState.spaceBetweenEchoesAtIdx(sampleIdx);
			float ampLossPerEcho = myState->delayState.amplitudeLostPerEchoAtIdx(sampleIdx);
			unsigned delayPerEchoInSamples = delayPerEcho*SAMPLE_RATE;
			for (unsigned echoIdx = 1; echoIdx <= MAX_DELAY_ECHOES; ++echoIdx) {
				unsigned curDelayIdx = echoIdx * delayPerEchoInSamples;
				float curAmp = max(0.f, 1.f - echoIdx*ampLossPerEcho);
				//if (curAmp <= 0.f) { break; }
				reduceDelayOutputs(voiceState, partialIdx, baseIdx + sampleIdx + curDelayIdx, curAmp*outputL, curAmp*outputR);
			}
		}
		updateVoiceParametersIfNeeded(voiceState, voiceNum, partialIdx);
		// TODO: use a proper reduction algorithm to determine when the note is complete
		if (partialIdx == NUM_PARTIALS-1 && !myState->volumeEnvelope.isActiveAtEndOfBlock()) {
			// signal no more samples
			unsigned bufferEndIdx = NUM_CH * ((baseIdx + (BUFFER_BLOCK_SIZE - 1)) % CIRCULAR_BUFFER_LEN);
			voiceState->sampleBuffer[bufferEndIdx] = NAN;
		}
	}

	__global__ void evaluateSynthVoiceBlockKernel(SynthState *synthState, unsigned voiceNum, unsigned baseIdx, unsigned samplesPerThread, float fundamentalFreq, bool released) {
	    unsigned partialNum = threadIdx.x;
		unsigned threadIdWithinPartial = blockIdx.x;
		computePartialOutput(synthState, voiceNum, baseIdx, partialNum, samplesPerThread, threadIdWithinPartial, fundamentalFreq, released);
	}

	__host__ void evaluateSynthVoiceBlockOnCpu(float bufferB[BUFFER_BLOCK_SIZE*NUM_CH], unsigned voiceNum, unsigned sampleIdx, float fundamentalFreq, bool released) {
		// need to obtain a lock on the synth state
		std::unique_lock<std::mutex> stateLock(synthStateMutex);
		// move pointer to d_synthState into a local for easy debugging
		SynthState *synthState = d_synthState;
		int threadsPerPartial = numThreadsPerPartial();
		int samplesPerThread = BUFFER_BLOCK_SIZE / threadsPerPartial;
		for (int partialIdx = 0; partialIdx < NUM_PARTIALS; ++partialIdx) {
			for (int threadIdWithinPartial = 0; threadIdWithinPartial < threadsPerPartial; ++threadIdWithinPartial) {
				computePartialOutput(synthState, voiceNum, sampleIdx, partialIdx, samplesPerThread, threadIdWithinPartial, fundamentalFreq, released);
			}
		}
		unsigned bufferStartIdx = NUM_CH * (sampleIdx % CIRCULAR_BUFFER_LEN);
		memcpy(bufferB, &synthState->voiceStates[voiceNum].sampleBuffer[bufferStartIdx], BUFFER_BLOCK_SIZE*NUM_CH*sizeof(float));
	}

	__host__ void evaluateSynthVoiceBlockCuda(float bufferB[BUFFER_BLOCK_SIZE*NUM_CH], unsigned voiceNum, unsigned sampleIdx, float fundamentalFreq, bool released) {
		unsigned threadsPerPartial = numThreadsPerPartial();
		unsigned samplesPerThread = BUFFER_BLOCK_SIZE / threadsPerPartial;
		evaluateSynthVoiceBlockKernel << <threadsPerPartial, NUM_PARTIALS >> >(d_synthState, voiceNum, sampleIdx, samplesPerThread, fundamentalFreq, released);

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

		newParameters->incrUUID();
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
		int threadsPerPartial = numThreadsPerPartial();
		PartialState *partialStates = new PartialState[threadsPerPartial*NUM_PARTIALS];
		for (int t = 0; t < threadsPerPartial; ++t) {
			for (int i = 0; i < NUM_PARTIALS; ++i) {
				partialStates[t*i] = PartialState(d_synthState, voiceNum, i);
			}
		}
		memcpyHostToSynthState(&d_synthState->voiceStates[voiceNum].partialStates, partialStates, sizeof(PartialState)*threadsPerPartial*NUM_PARTIALS);
		memsetSynthState(&d_synthState->voiceStates[voiceNum].sampleBuffer, 0, CIRCULAR_BUFFER_LEN*NUM_CH*sizeof(float));
		delete partialStates;
	}

}