#ifndef ADSREDITOR_H
#define ADSREDITOR_H

#include "../JuceLibraryCode/JuceHeader.h"
#include "ParameterEditor.h"

class PluginEditor;

class ADSREditor : public ParameterEditor
{
	PluginEditor *editor;
	ADSR *adsr;
public:
	ADSREditor(PluginEditor *editor, ADSR *adsr);
	~ADSREditor();
	void onParameterChanged(int parameterNum, float value) override;
};

#endif
