#include "ADSRComponent.h"

#include "PluginEditor.h"

ADSRComponent::ADSRComponent(JuceDemoPluginAudioProcessorEditor *editor, ADSR *adsr)
	: editor(editor), adsr(adsr),
	  attackSlider (Slider::Rotary, Slider::NoTextBox),
	  decaySlider  (Slider::Rotary, Slider::NoTextBox),
	  sustainSlider(Slider::Rotary, Slider::NoTextBox),
	  releaseSlider(Slider::Rotary, Slider::NoTextBox),
	  stretchSlider(Slider::Rotary, Slider::NoTextBox)
{
	Slider* sliders[5] = { &attackSlider, &decaySlider, &sustainSlider, &releaseSlider, &stretchSlider };
	float lowerBounds[5] = { 0, 0, 0, 0, 0 };
	float upperBounds[5] = { 2, 2, 1, 2, 5 };
	for (int i = 0; i < sizeof(sliders)/sizeof(Slider*); ++i) {
		addAndMakeVisible(sliders[i]);
		sliders[i]->addListener(this);
		sliders[i]->setRange(lowerBounds[i], upperBounds[i], 0.0);
	}
}


ADSRComponent::~ADSRComponent()
{
}

void ADSRComponent::resized()
{
	int sliderSize = 36;
	int sliderPadding = 40;
	attackSlider.setBounds (sliderPadding * 0, 0, sliderSize, sliderSize);
	decaySlider.setBounds  (sliderPadding * 1, 0, sliderSize, sliderSize);
	sustainSlider.setBounds(sliderPadding * 2, 0, sliderSize, sliderSize);
	releaseSlider.setBounds(sliderPadding * 3, 0, sliderSize, sliderSize);
	stretchSlider.setBounds(sliderPadding * 4, 0, sliderSize, sliderSize);
}

void ADSRComponent::sliderValueChanged(Slider *slider) {
	float value = slider->getValue();
	if (slider == &attackSlider) {
		adsr->setAttack(value);
	} else if (slider == &decaySlider) {
		adsr->setDecay(value);
	} else if (slider == &sustainSlider) {
		adsr->setSustain(value);
	} else if (slider == &releaseSlider) {
		adsr->setRelease(value);
	} else if (slider == &stretchSlider) {
		adsr->setScaleByPartialIdx(value);
	}
	editor->parametersChanged();
}