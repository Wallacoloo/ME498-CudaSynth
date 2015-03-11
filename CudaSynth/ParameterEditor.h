#ifndef PARAMETEREDITOR_H
#define PARAMETEREDITOR_H

#include "JuceLibraryCode/JuceHeader.h"
#include "kernel.h"

class PluginEditor;

class ParameterEditor :
	public Component, public SliderListener
{
	PluginEditor *editor;
	Label editorLabel;
	int nSliders;
	Slider *sliders;
	Label *labels;
	// maps the slider index i to the parameter index j
	// i.e. j = parameterIndexMap[i]
	const int *parameterIndexMap;
	void paint(Graphics&) override;
public:
	ParameterEditor(PluginEditor *editor, const char* editorLabel, const char** labelNames, const char** tooltips, const float parameterBounds[][2], const int* usableParameterIndices);
	~ParameterEditor();
	void sliderValueChanged(Slider*) override;
	void resized() override;
	void refreshSliderValues();
	virtual void onParameterChanged(int parameterNum, float value) = 0;
	virtual float getParameterValue(int parameterNum) const = 0;
};

#endif
