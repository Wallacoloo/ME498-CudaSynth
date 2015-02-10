#ifndef ADSRCOMPONENT_H
#define ADSRCOMPONENT_H

#include "../JuceLibraryCode/JuceHeader.h"
#include "kernel.h"

class JuceDemoPluginAudioProcessorEditor;

class ADSRComponent :
	public Component, public SliderListener
{
	JuceDemoPluginAudioProcessorEditor *editor;
	ADSR *adsr;
	Slider attackSlider, decaySlider, sustainSlider, releaseSlider, stretchSlider;
public:
	ADSRComponent(JuceDemoPluginAudioProcessorEditor *editor, ADSR *adsr);
	~ADSRComponent();
	void sliderValueChanged(Slider*) override;
	void resized() override;
};

#endif
