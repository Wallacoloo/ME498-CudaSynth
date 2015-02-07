#include "ADSRComponent.h"

#include "PluginEditor.h"

ADSRComponent::ADSRComponent(JuceDemoPluginAudioProcessorEditor *editor, ADSR *adsr)
	: editor(editor), adsr(adsr),
	  attackSlider (Slider::Rotary, Slider::NoTextBox),
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
	int sliderSize = 36;
	int sliderPadding = 40;
	attackSlider.setBounds (sliderPadding * 0, 0, sliderSize, sliderSize);
	decaySlider.setBounds  (sliderPadding * 1, 0, sliderSize, sliderSize);
	sustainSlider.setBounds(sliderPadding * 2, 0, sliderSize, sliderSize);
	releaseSlider.setBounds(sliderPadding * 3, 0, sliderSize, sliderSize);
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
	}
	editor->parametersChanged();
}