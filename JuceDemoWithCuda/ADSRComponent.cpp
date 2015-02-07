#include "ADSRComponent.h"


ADSRComponent::ADSRComponent()
	: attackSlider (Slider::Rotary, Slider::NoTextBox),
	  decaySlider  (Slider::Rotary, Slider::NoTextBox),
	  sustainSlider(Slider::Rotary, Slider::NoTextBox),
	  releaseSlider(Slider::Rotary, Slider::NoTextBox)
{
	Slider* sliders[4] = { &attackSlider, &decaySlider, &sustainSlider, &releaseSlider };
	for (int i = 0; i < 4; ++i) {
		addAndMakeVisible(sliders[i]);
		sliders[i]->addListener(this);
		sliders[i]->setRange(0.0, 1.0, 0.01);
	}
}


ADSRComponent::~ADSRComponent()
{
}

void ADSRComponent::resized()
{
	attackSlider.setBounds (0,   0, 36, 36);
	decaySlider.setBounds  (40,  0, 36, 36);
	sustainSlider.setBounds(80,  0, 36, 36);
	releaseSlider.setBounds(120, 0, 36, 36);
}

void ADSRComponent::sliderValueChanged(Slider *slider) {

}