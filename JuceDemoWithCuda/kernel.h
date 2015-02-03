#ifndef KERNEL_H
#define KERNEL_H

#include "defines.h"

namespace kernel {
	// Struct to hold ALL parameter states at a single instant in time.
	// There will be two of these sent during each synthesis block:
	//   1 for at the start of the block,
	//   another for at the end of the block.
	//   The actual value at any time will be the linear interpolation of the two.
	struct ParameterStates {
		// hand-drawn partial envelopes
		float partialLevels[NUM_PARTIALS];
	};

	// Call to evaluate the next N samples of a synthesizer voice into bufferB.
	void evaluateSynthVoiceBlock(float *bufferB, unsigned baseIdx, float fundamentalFreq);

	// Call whenever the user edits one of the synth parameters
	void parameterStatesChanged(const ParameterStates *newParameters);
}

using namespace kernel;
#endif