#ifndef LFOEDITOR_H
#define LFOEDITOR_H

#include "../JuceLibraryCode/JuceHeader.h"
#include "ParameterEditor.h"

class PluginEditor;

class LFOEditor : public ParameterEditor
{
	LFO *lfo;
public:
	LFOEditor(PluginEditor *editor, LFO *lfo);
	~LFOEditor();
	void onParameterChanged(int parameterNum, float value) override;
};

#endif
