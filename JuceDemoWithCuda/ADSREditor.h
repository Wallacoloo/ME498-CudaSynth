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
	enum ADSREditorOptions {
		Classic,      // attack (startLevel=0.0), (peak=1.0), decay, sustain, release, (endLevel=0.0), stretchByFreq
		AsrWithPeaks, // attack, startLevel, (no decay or peak), sustain, release, endLevel, stretchByFreq
	};
	ADSREditor(PluginEditor *editor, ADSR *adsr, const char* editorLabel, ADSREditorOptions opt);
	~ADSREditor();
	void onParameterChanged(int parameterNum, float value) override;
};

#endif
