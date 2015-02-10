#ifndef ENVELOPEEDITOR_H
#define ENVELOPEEDITOR_H

#include "../JuceLibraryCode/JuceHeader.h"
#include "kernel.h"

class PluginEditor;

class EnvelopeEditor :
	public Component, public SliderListener
{
	PluginEditor *editor;
	ADSR *adsr;
	Slider attackSlider, decaySlider, sustainSlider, releaseSlider, stretchSlider;
	Label attackLabel, decayLabel, sustainLabel, releaseLabel, stretchLabel;
public:
	EnvelopeEditor(PluginEditor *editor, ADSR *adsr);
	~EnvelopeEditor();
	void sliderValueChanged(Slider*) override;
	void resized() override;
};

#endif
