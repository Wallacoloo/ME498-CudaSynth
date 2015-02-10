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
	//ADSR *adsr;
	//Slider attackSlider, decaySlider, sustainSlider, releaseSlider, stretchSlider;
	//Label attackLabel, decayLabel, sustainLabel, releaseLabel, stretchLabel;
public:
	ParameterEditor(PluginEditor *editor, int nSliders, const char** labelNames);
	~ParameterEditor();
	void sliderValueChanged(Slider*) override;
	void resized() override;
	virtual void onParameterChanged(int parameterNum, float value) = 0;
};

#endif
