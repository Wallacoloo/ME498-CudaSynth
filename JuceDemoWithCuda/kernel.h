#ifndef KERNEL_H
#define KERNEL_H

#include "defines.h"

namespace kernel {

	class ADSR {
		// level at t=0
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
		float scaleByPartialIdx;
	public:
		ADSR()
		: startLevel(0), a(0), peakLevel(1), d(0), s(0), r(0), releaseLevel(0), scaleByPartialIdx(0) {}
		inline void setStartLevel(float level) {
			this->startLevel = level;
		}
		inline void setAttack(float attack) {
			this->a = attack;
		}
		inline void setPeakLevel(float level) {
			this->peakLevel = level;
		}
		inline void setDecay(float decay) {
			this->d = decay;
		}
		inline void setSustain(float sustain) {
			this->s = sustain;
		}
		inline void setRelease(float release) {
			this->r = release;
		}
		inline void setReleaseLevel(float level) {
			this->releaseLevel = level;
		}
		inline void setScaleByPartialIdx(float s) {
			this->scaleByPartialIdx = s;
		}
		inline HOST DEVICE float getStartLevel() const {
			return startLevel;
		}
		inline HOST DEVICE float getAttack() const {
			return a;
		}
		inline HOST DEVICE float getAttackFor(unsigned partialIdx) const {
			return a * getTimeScaleFor(partialIdx);
		}
		inline HOST DEVICE float getPeakLevel() const {
			return peakLevel;
		}
		inline HOST DEVICE float getDecay() const {
			return d;
		}
		inline HOST DEVICE float getDecayFor(unsigned partialIdx) const {
			return d * getTimeScaleFor(partialIdx);
		}
		inline HOST DEVICE float getSustain() const {
			return s;
		}
		inline HOST DEVICE float getRelease() const {
			return r;
		}
		inline HOST DEVICE float getReleaseFor(unsigned partialIdx) const {
			return r * getTimeScaleFor(partialIdx);
		}
		inline HOST DEVICE float getReleaseLevel() const {
			return releaseLevel;
		}
		inline HOST DEVICE float getScaleByPartialIdx() const {
			return scaleByPartialIdx;
		}
		inline HOST DEVICE float getTimeScaleFor(unsigned partialIdx) const {
			return 1.f + scaleByPartialIdx*partialIdx/NUM_PARTIALS;
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
		inline HOST DEVICE ADSRLFOEnvelope* getAdsrLfo() {
			return &adsrLfo;
		}
	};

	// Struct to hold ALL parameter states at a single instant in time.
	// There will be two of these sent during each synthesis block:
	//   1 for at the start of the block,
	//   another for at the end of the block.
	//   The actual value at any time will be the linear interpolation of the two.
	struct ParameterStates {
		// hand-drawn partial envelopes
		float partialLevels[NUM_PARTIALS];
		ADSRLFOEnvelope volumeEnvelope;
		ADSRLFOEnvelope stereoPanEnvelope;
		DetuneEnvelope detuneEnvelope;
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