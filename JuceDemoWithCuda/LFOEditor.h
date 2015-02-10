#ifndef ENVELOPEEDITOR_H
#define LFOEDITOR_H

#include "../JuceLibraryCode/JuceHeader.h"
#include "kernel.h"

class PluginEditor;

class LFOEditor :
	public Component, public SliderListener
{
	PluginEditor *editor;
	LFO *lfo;
	Slider attackSlider, decaySlider, sustainSlider, releaseSlider, stretchSlider;
	Label attackLabel, decayLabel, sustainLabel, releaseLabel, stretchLabel;
public:
	LFOEditor(PluginEditor *editor, LFO *lfo);
	~LFOEditor();
	void sliderValueChanged(Slider*) override;
	void resized() override;
};

#endif
