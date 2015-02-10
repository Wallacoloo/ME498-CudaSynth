#include "LFOEditor.h"

#include "PluginEditor.h"

LFOEditor::LFOEditor(PluginEditor *editor, LFO *lfo)
	: editor(editor), lfo(lfo),
	attackSlider(Slider::Rotary, Slider::NoTextBox),
	decaySlider(Slider::Rotary, Slider::NoTextBox),
	sustainSlider(Slider::Rotary, Slider::NoTextBox),
	releaseSlider(Slider::Rotary, Slider::NoTextBox),
	stretchSlider(Slider::Rotary, Slider::NoTextBox),
	attackLabel("Attack", "A"),
	decayLabel("Decay", "D"),
	sustainLabel("Sustain", "S"),
	releaseLabel("Release", "R"),
	stretchLabel("Stretch", "Stretch")
{
	Slider* sliders[5] = { &attackSlider, &decaySlider, &sustainSlider, &releaseSlider, &stretchSlider };
	Label *labels[5] = { &attackLabel, &decayLabel, &sustainLabel, &releaseLabel, &stretchLabel };
	float lowerBounds[5] = { 0, 0, 0, 0, -0.8 };
	float upperBounds[5] = { 2, 2, 1, 2, 5 };
	for (int i = 0; i < sizeof(sliders) / sizeof(Slider*); ++i) {
		// configure slider
		addAndMakeVisible(sliders[i]);
		sliders[i]->addListener(this);
		sliders[i]->setRange(lowerBounds[i], upperBounds[i], 0.0);
		// configure slider's label
		addAndMakeVisible(labels[i]);
		// labels[i]->attachToComponent(sliders[i], false);
		labels[i]->setFont(Font(11.f));
	}
}


LFOEditor::~LFOEditor()
{
}

void LFOEditor::resized()
{
	int sliderSize = 36;
	int sliderPadding = 40;
	int labelHeight = 16;
	Slider* sliders[5] = { &attackSlider, &decaySlider, &sustainSlider, &releaseSlider, &stretchSlider };
	Label *labels[5] = { &attackLabel, &decayLabel, &sustainLabel, &releaseLabel, &stretchLabel };
	for (int i = 0; i < sizeof(sliders) / sizeof(Slider*); ++i) {
		sliders[i]->setBounds(sliderPadding*i, 0, sliderSize, sliderSize);
		labels[i]->setBounds(sliderPadding*i, sliderSize, sliderSize, labelHeight);
	}
}

void LFOEditor::sliderValueChanged(Slider *slider) {
	/*float value = slider->getValue();
	if (slider == &attackSlider) {
		adsr->setAttack(value);
	}
	else if (slider == &decaySlider) {
		adsr->setDecay(value);
	}
	else if (slider == &sustainSlider) {
		adsr->setSustain(value);
	}
	else if (slider == &releaseSlider) {
		adsr->setRelease(value);
	}
	else if (slider == &stretchSlider) {
		adsr->setScaleByPartialIdx(value);
	}
	editor->parametersChanged();*/
}