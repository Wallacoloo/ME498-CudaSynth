#ifndef PARAMETEREDITOR_H
#define PARAMETEREDITOR_H

#include "../JuceLibraryCode/JuceHeader.h"
#include "kernel.h"

class PluginEditor;

class ParameterEditor :
	public Component, public SliderListener
{
	PluginEditor *editor;
	int nSliders;
	Slider *sliders;
	Label *labels;
	// maps the slider index i to the parameter index j
	// i.e. j = parameterIndexMap[i]
	const int *parameterIndexMap;
public:
	ParameterEditor(PluginEditor *editor, const char** labelNames, const float parameterBounds[][2], const int* usableParameterIndices);
	~ParameterEditor();
	void sliderValueChanged(Slider*) override;
	void resized() override;
	virtual void onParameterChanged(int parameterNum, float value) = 0;
};

#endif
