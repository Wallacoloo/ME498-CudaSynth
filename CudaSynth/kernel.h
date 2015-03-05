#ifndef KERNEL_H
#define KERNEL_H

#include "defines.h"
#include <algorithm> //for std::max

// Minimum length for each portion of the ADSR envelope
// Mins may be needed to avoid divisions by zero, etc.
// #define MIN_ADSR_SEGMENT_LENGTH_SAMPLES BUFFER_BLOCK_SIZE
#define MIN_ADSR_SEGMENT_LENGTH_SAMPLES 2
// certain logic in the kernel may require that each segment have non-zero slope
// #define ADSR_MIN_SEGMENT_VALUE_DIFF 0.0000001f
#define ADSR_MIN_SEGMENT_VALUE_DIFF 0
// sustain, end mode segments have finited length. Choose something long, but not so long that rounding errors arise
#define ADSR_LONG_SEGMENT_LENGTH 4096.f


namespace kernel {


	class PiecewiseFunction {
		// if a point has level=NAN, it should be considered as non-existent
		// non-existent points should only be located at the END of the array
		struct Point {
			float time, level;
		};
		Point pieces[PIECEWISE_MAX_PIECES];
	public:
		PiecewiseFunction() {
			for (int i = 0; i < PIECEWISE_MAX_PIECES; ++i) {
				pieces[i].time = 0;
				pieces[i].level = NAN;
			}
		}
		float startTimeOfPiece(int pieceNum) const {
			return pieces[pieceNum].time;
		}
		float startLevelOfPiece(int pieceNum) const {
			return pieces[pieceNum].level;
		}
		unsigned numPoints() const {
			for (int p = 0; p < PIECEWISE_MAX_PIECES; ++p) {
				if (std::isnan(pieces[p].level)) {
					return p;
				}
			}
			return PIECEWISE_MAX_PIECES;
		}
		// shift the point and all following points such that:
		//   this point starts at the absolute time newStartTime
		//   and has level newLevel
		// return the actual startTime (e.g. if we try to set a negative length)
		float movePoint(int pointNum, float newStartTime, float newLevel) {
			if (pointNum == 0) {
				newStartTime = 0.f;
			} else {
				newStartTime = std::max(startTimeOfPiece(pointNum-1), newStartTime);
			}
			pieces[pointNum].level = newLevel;
			float shiftAmt = newStartTime - pieces[pointNum].time;
			for (int p = pointNum; p < numPoints(); ++p) {
				pieces[p].time += shiftAmt;
			}
			return newStartTime;
		}
		// insert a new point with given time & level.
		// return the index of said point
		int insertPoint(float newStartTime, float newLevel) {
			int nextPoint = numPoints();
			// check if we can allocate a new point
			if (nextPoint == PIECEWISE_MAX_PIECES) {
				return -1;
			}
			// determine where to insert the point
			int maxPBefore;
			for (maxPBefore = 0; maxPBefore < nextPoint-1; ++maxPBefore) {
				if (pieces[maxPBefore+1].time > newStartTime) {
					break;
				}
			}
			// insert the point by swapping our data with maxPBefore+1,
			// and then pushing those changes down the line
			float lastPointTime = newStartTime;
			float lastPointLevel = newLevel;
			for (int p = maxPBefore + 1; p <= nextPoint; ++p) {
				std::swap(pieces[p].time, lastPointTime);
				std::swap(pieces[p].level, lastPointLevel);
			}
			// return index of the point we inserted
			return maxPBefore + 1;
		}
		void removePoint(int pointNum) {
			int newNumPoints = numPoints() - 1;
			// shift all points up
			for (int p = pointNum; p < newNumPoints; ++p) {
				std::swap(pieces[p].time, pieces[p + 1].time);
				std::swap(pieces[p].level, pieces[p + 1].level);
			}
			// delete the previous point
			pieces[newNumPoints].level = NAN;
		}
	};

	class ADSR {
	public:
		enum Mode {
			AttackMode = 0,
			DecayMode = 1,
			SustainMode = 2,
			ReleaseMode = 3,
			EndMode = 4, // need padding for when trying to access past the end of the envelope
			PastEndMode = 5,
		};
	private:
		// essentially have 4 line segments:
		// attack, decay, sustain, release
		// each segment has a starting value and a length.
		// release mode also has an ending value, which we accomplish with a 5th & 6th (constant) segment
		// sustain has infinite length, everything else has finite length.
		float levelsAndLengths[6][2];
		float scaleByPartialIdx;
		float amplifyByPartialIdx;
		/*// level at t=0
		float startLevel;
		// time it takes for the signal to rise to its peak
		float a;
		float peakLevel;
		// time it takes for the signal to decay to the sustain level
		float d;
		// sustain amplitude
		float s;
		// once note is released, time it takes for signal to decay to its release level.
		float r;
		// level at t=infinity. Usually 0 when the envelope represents amplitude, but has uses in detune/filter envelopes
		float releaseLevel;
		// used to (externally) multiply the a, d and r values by (1+scaleByPartialIdx*p)
		float scaleByPartialIdx;*/
	public:
		ADSR() {
			// Sustain, EndMode must still have a finite length; set it to something absurdly long
			// Don't make it too lengthy though, or rounding errors may occur in future parts of the kernel.
			float longDuration = ADSR_LONG_SEGMENT_LENGTH;
			// set start level to 0 & attack to 0
			setStartLevel(0.f);
			setAttack(0.f);
			// set peak to 1 & decay to 0
			// setSegmentStartLevel(DecayMode, 1.f);
			setPeakLevel(1.f);
			setDecay(0.f);
			// set length sustain mode to effective infinity
			setSegmentLength(SustainMode, longDuration);
			// set sustain level to 1.0
			setSustain(1.f);
			// set release level & length to 0
			setReleaseLevel(0.f);
			setRelease(0.f);
			// set length of EndMode & PastEndMode to effective infinity
			setSegmentLength(EndMode, longDuration);
			setSegmentLength(PastEndMode, longDuration);
			// default to no scaling by partial index.
			setScaleByPartialIdx(0);
			setAmplificationByPartialIdx(0);
		}
		inline void setSegmentStartLevel(Mode mode, float value) {
			// certain logic in the kernel may require that each segment have non-zero slope
			if (ADSR_MIN_SEGMENT_VALUE_DIFF != 0) {
				if (mode != AttackMode && levelsAndLengths[(unsigned)mode - 1][0] == value) {
					setSegmentStartLevel(mode, value + ADSR_MIN_SEGMENT_VALUE_DIFF);
					return;
				} else if (mode != PastEndMode && levelsAndLengths[(unsigned)mode + 1][0] == value) {
					setSegmentStartLevel(mode, value + ADSR_MIN_SEGMENT_VALUE_DIFF);
					return;
				}
			}
			levelsAndLengths[(unsigned)mode][0] = value;
		}
		inline void setSegmentLength(Mode mode, float value) {
			// certain logic in the kernel may require that each segment have finite slope
			//   or a minimum length
			value = std::max(value, INV_SAMPLE_RATE*MIN_ADSR_SEGMENT_LENGTH_SAMPLES);
			levelsAndLengths[(unsigned)mode][1] = value;
		}
		inline HOST DEVICE float getSegmentStartLevel(Mode mode, unsigned partialIdx=0) const {
			return levelsAndLengths[(unsigned)mode][0] * getAmplificationFor(partialIdx);
		}
		inline HOST DEVICE float getSegmentLength(Mode mode, unsigned partialIdx=0) const {
			return levelsAndLengths[(unsigned)mode][1] * getTimeScaleFor(partialIdx);
		}
		inline HOST DEVICE float getSegmentSlope(Mode mode, unsigned partialIdx = 0) const {
			float y0 =  getSegmentStartLevel(mode);
			float y1 = getSegmentStartLevel((Mode)((unsigned)mode + 1));
			float length = getSegmentLength(mode, partialIdx);
			return (y1 - y0) / length;
		}
		inline void setStartLevel(float level) {
			setSegmentStartLevel(AttackMode, level);
		}
		inline void setAttack(float attack) {
			setSegmentLength(AttackMode, attack);
		}
		inline void setPeakLevel(float level) {
			setSegmentStartLevel(DecayMode, level);
		}
		inline void setDecay(float decay) {
			setSegmentLength(DecayMode, decay);
		}
		inline void setSustain(float level) {
			setSegmentStartLevel(SustainMode, level);
			setSegmentStartLevel(ReleaseMode, level);
		}
		inline void setRelease(float release) {
			setSegmentLength(ReleaseMode, release);
		}
		inline void setReleaseLevel(float level) {
			setSegmentStartLevel(EndMode, level);
			setSegmentStartLevel(PastEndMode, level);
		}
		inline void setScaleByPartialIdx(float s) {
			this->scaleByPartialIdx = s;
		}
		inline void setAmplificationByPartialIdx(float a) {
			this->amplifyByPartialIdx = a;
		}
		inline HOST DEVICE float getStartLevel() const {
			return getSegmentStartLevel(AttackMode);
		}
		inline HOST DEVICE float getAttack() const {
			return getSegmentLength(AttackMode);
		}
		inline HOST DEVICE float getPeakLevel() const {
			return getSegmentStartLevel(DecayMode);
		}
		inline HOST DEVICE float getDecay() const {
			return getSegmentLength(DecayMode);
		}
		inline HOST DEVICE float getSustain() const {
			return getSegmentStartLevel(SustainMode);
		}
		inline HOST DEVICE float getRelease() const {
			return getSegmentLength(ReleaseMode);
		}
		inline HOST DEVICE float getReleaseLevel() const {
			return getSegmentStartLevel(EndMode);
		}
		inline HOST DEVICE float getScaleByPartialIdx() const {
			return scaleByPartialIdx;
		}
		inline HOST DEVICE float getAmplificationByPartialIdx() const {
			return amplifyByPartialIdx;
		}
		inline HOST DEVICE float getTimeScaleFor(unsigned partialIdx) const {
			return 1.f + scaleByPartialIdx*partialIdx/NUM_PARTIALS;
		}
		inline HOST DEVICE float  getAmplificationFor(unsigned partialIdx) const {
			return 1.f + amplifyByPartialIdx*partialIdx / NUM_PARTIALS;
		}
	};

	class LFO {
		ADSR lfoFreq;
		ADSR lfoDepth;
	public:
		LFO() : lfoFreq(), lfoDepth() {}
		inline HOST DEVICE const ADSR* getFreqAdsr() const {
			return &lfoFreq;
		}
		inline HOST DEVICE ADSR* getFreqAdsr() {
			return &lfoFreq;
		}
		inline HOST DEVICE const ADSR* getDepthAdsr() const {
			return &lfoDepth;
		}
		inline HOST DEVICE ADSR* getDepthAdsr() {
			return &lfoDepth;
		}
	};

	class ADSRLFOEnvelope {
		ADSR adsr;
		LFO lfo;
	public:
		inline HOST DEVICE ADSR* getAdsr() {
			return &adsr;
		}
		inline HOST DEVICE LFO* getLfo() {
			return &lfo;
		}
	};

	class DetuneEnvelope {
		float randSeed;
		// amount of randomness to be mixed in
		float randMix;
		ADSRLFOEnvelope adsrLfo;
	public:
		DetuneEnvelope() {
			randSeed = 0;
			randMix = 0;
		}
		inline HOST DEVICE ADSRLFOEnvelope* getAdsrLfo() {
			return &adsrLfo;
		}
		inline HOST DEVICE float getRandSeed() const {
			return randSeed;
		}
		inline HOST DEVICE float getRandMix() const {
			return randMix;
		}
		inline void setRandSeed(float s) {
			this->randSeed = s;
		}
		inline void setRandMix(float m) {
			this->randMix = m;
		}
	};

	class DelayEnvelope {
		ADSRLFOEnvelope spaceBetweenEchoes;
		ADSRLFOEnvelope amplitudeLostPerEcho;
	public:
		inline HOST DEVICE ADSRLFOEnvelope* getSpaceBetweenEchoes() {
			return &spaceBetweenEchoes;
		}
		inline HOST DEVICE ADSRLFOEnvelope* getAmplitudeLostPerEcho() {
			return &amplitudeLostPerEcho;
		}
	};

	class FilterEnvelope {
		PiecewiseFunction shape;
	public:
		FilterEnvelope() {
			shape.movePoint(0, 0, 0);
			shape.movePoint(1, 0, 1);
			shape.movePoint(2, NYQUIST_RATE_RAD, 1);
			shape.movePoint(3, NYQUIST_RATE_RAD, 0);
		}
		HOST DEVICE PiecewiseFunction* getShape() {
			return &shape;
		}
	};

	// Struct to hold ALL parameter states at a single instant in time.
	// There will be two of these sent during each synthesis block:
	//   1 for at the start of the block,
	//   another for at the end of the block.
	//   The actual value at any time will be the linear interpolation of the two.
	struct ParameterStates {
		static unsigned nextUUID;
		mutable unsigned UUID;
		// hand-drawn partial envelopes
		float partialLevels[NUM_PARTIALS];
		ADSRLFOEnvelope volumeEnvelope;
		ADSRLFOEnvelope stereoPanEnvelope;
		DetuneEnvelope detuneEnvelope;
		DelayEnvelope delayEnvelope;
		FilterEnvelope filterEnvelope;
		ParameterStates() {
			UUID = nextUUID++;
			// initialize partials to uniform level
			for (int p = 0; p < NUM_PARTIALS; ++p) {
				partialLevels[p] = 0.5;
			}
			// default to no volume LFO
			volumeEnvelope.getLfo()->getDepthAdsr()->setSustain(0.f);
			// default to no panning
			stereoPanEnvelope.getAdsr()->setSustain(0.f);
			stereoPanEnvelope.getLfo()->getDepthAdsr()->setSustain(0.f);
			// default to no detuning
			detuneEnvelope.getAdsrLfo()->getAdsr()->setSustain(0.f);
			detuneEnvelope.getAdsrLfo()->getLfo()->getDepthAdsr()->setSustain(0.f);
			// default to no delays (taken care of)
			// delayEnvelope.getAmplitudeLostPerEcho()->getAdsr()->setSustain(1.f);
			// default to no delay LFO
			delayEnvelope.getAmplitudeLostPerEcho()->getLfo()->getDepthAdsr()->setSustain(0.f);
			delayEnvelope.getSpaceBetweenEchoes()->getLfo()->getDepthAdsr()->setSustain(0.f);
		}
		void incrUUID() const {
			UUID = nextUUID++;
		}
	};

	// Call to evaluate the next N samples of a synthesizer voice into bufferB.
	void evaluateSynthVoiceBlock(float *bufferB, unsigned voiceNum, unsigned baseIdx, float fundamentalFreq, bool released);

	// Call whenever the user edits one of the synth parameters
	void parameterStatesChanged(const ParameterStates *newParameters);

	// Call at the onset of a note BEFORE calculating the next block
	void onNoteStart(unsigned voiceNum);
}

using namespace kernel;
#endif