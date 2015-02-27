#ifndef ADSREDITOR_H
#define ADSREDITOR_H

#include "../JuceLibraryCode/JuceHeader.h"
#include "ParameterEditor.h"

class PluginEditor;

class ADSREditor : public ParameterEditor
{
	ADSR *adsr;
public:
	// the ADSR envelope can be presented in a few different ways
	enum KnobTypes {
		ClassicKnobs,      // attack (startLevel=0.0), (peak=1.0), decay, sustain, release, (endLevel=0.0), stretchByFreq
		ClassicKnobsWithScaleByIdx,
		AsrWithPeaksKnobs, // attack, startLevel, (no decay or peak), sustain, release, endLevel, stretchByFreq
	};
	enum KnobLimits {
		NormalizedDepthLimits, // ADSR should have bounds [0, 1]
		NormalizedDepthLimitsPlusOrMinus, // ADSR should have bounds [-1, 1]
		LFOFrequencyLimits, //ADSR should have bounds[0, frequency limit]
	};
	ADSREditor(PluginEditor *editor, ADSR *adsr, const char* editorLabel, KnobTypes knobTypes, KnobLimits knobLimits);
	~ADSREditor();
	void onParameterChanged(int parameterNum, float value) override;
	float getParameterValue(int parameterNum) const override;
};

#endif
