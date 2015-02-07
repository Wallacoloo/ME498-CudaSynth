#ifndef ADSRCOMPONENT_H
#define ADSRCOMPONENT_H

#include "../JuceLibraryCode/JuceHeader.h"

class ADSRComponent :
	public Component, public SliderListener
{
	Slider attackSlider, decaySlider, sustainSlider, releaseSlider;
public:
	ADSRComponent();
	~ADSRComponent();
	void sliderValueChanged(Slider*) override;
	void resized() override;
};

#endif
