#ifndef DETUNERANDEDITOR_H
#define DETUNERANDEDITOR_H

#include "ParameterEditor.h"
#include "kernel.h"

class DetuneRandEditor :
	public ParameterEditor
{
	DetuneEnvelope *detune;
public:
	DetuneRandEditor(PluginEditor *editor, DetuneEnvelope *detune, const char* editorLabel);
	~DetuneRandEditor();
	void onParameterChanged(int parameterNum, float value) override;
	float getParameterValue(int parameterNum) const override;
};


#endif