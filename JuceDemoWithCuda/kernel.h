#ifndef KERNEL_H
#define KERNEL_H

#include "defines.h"

// Struct to hold ALL parameter states at a single instant in time.
// There will be two of these sent during each synthesis block:
//   1 for at the start of the block,
//   another for at the end of the block.
//   The actual value at any time will be the linear interpolation of the two.
struct ParameterStates {
	// hand-drawn partial envelopes
	float partialLevels[NUM_PARTIALS];
};

void fillSineWaveVoice(float *bufferB, unsigned baseIdx, float fundamentalFreq, const ParameterStates *newParameters=NULL);

#endif