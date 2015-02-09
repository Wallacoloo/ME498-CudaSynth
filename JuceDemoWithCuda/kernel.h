#ifndef KERNEL_H
#define KERNEL_H

#include "defines.h"

namespace kernel {

	class ADSR {
		// time it takes for the signal to rise to 1.0
		float a;
		// time it takes for the signal to decay to the sustain level
		float d;
		// sustain amplitude
		float s;
		// once note is released, time it takes for signal to decay to 0.
		float r;
	public:
		ADSR(float a = 0.f, float d = 0.f, float s = 0.f, float r = 0.f) : a(a), d(d), s(s), r(r) {}
		inline void setAttack(float attack) {
			this->a = attack;
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
		inline HOST DEVICE float getAttack() const {
			return a;
		}
		inline HOST DEVICE float getDecay() const {
			return d;
		}
		inline HOST DEVICE float getSustain() const {
			return s;
		}
		inline HOST DEVICE float getRelease() const {
			return r;
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
		ADSR volumeEnvelope;
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