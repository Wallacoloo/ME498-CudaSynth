#ifndef KERNEL_H
#define KERNEL_H

#include "defines.h"

namespace kernel {

	class ADSR {
		enum Mode {
			AttackMode = 0,
			DecayMode = 1,
			SustainMode = 2,
			ReleaseMode = 3,
			EndMode = 4,
		};
		// essentially have 4 line segments:
		// attack, decay, sustain, release
		// each segment has a starting value and a length.
		// release mode also has an ending value, which we accomplish with a 5th (constant) segment
		// sustain has infinite length, everything else has finite length.
		float levelsAndLengths[5][2];
		float scaleByPartialIdx;
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
			for (int i = 0; i < 5; ++i) {
				levelsAndLengths[i][0] = 0;
				levelsAndLengths[i][1] = 0;
			}
			// set peak to 1
			levelsAndLengths[DecayMode][0] = 1;
			// set lengths of sustain mode and end mode to infinity
			levelsAndLengths[SustainMode][1] = INFINITY;
			levelsAndLengths[EndMode][1] = INFINITY;
		}
		inline void setSegmentStartLevel(Mode mode, float value) {
			levelsAndLengths[(unsigned)mode][0] = value;
		}
		inline void setSegmentLength(Mode mode, float value) {
			levelsAndLengths[(unsigned)mode][1] = value;
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
			setSegmentLength(AttackMode, decay);
		}
		inline void setSustain(float level) {
			setSegmentStartLevel(SustainMode, level);
			setSegmentStartLevel(ReleaseMode, level);
		}
		inline void setRelease(float release) {
			setSegmentLength(AttackMode, release);
		}
		inline void setReleaseLevel(float level) {
			setSegmentStartLevel(EndMode, level);
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