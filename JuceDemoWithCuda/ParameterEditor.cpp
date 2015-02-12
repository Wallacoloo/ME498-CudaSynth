#include "ParameterEditor.h"
#include "PluginEditor.h"

ParameterEditor::ParameterEditor(PluginEditor *editor, int nSliders, const char** labelNames)
	: editor(editor), nSliders(nSliders), sliders(new Slider[nSliders]), labels(new Label[nSliders])
{
	for (int i = 0; i < nSliders; ++i) {
		sliders[i].setSliderStyle(Slider::Rotary);
		sliders[i].setTextBoxStyle(Slider::NoTextBox, false, 0, 0);
		addAndMakeVisible(sliders[i]);
		sliders[i].addListener(this);
		sliders[i].setRange(0.f, 1.f, 0.f);

		addAndMakeVisible(labels[i]);
		labels[i].setFont(Font(11.f));
		labels[i].setText(labelNames[i], sendNotification);
	}
	// call resized() to compute the desired bounds
	resized();
}


ParameterEditor::~ParameterEditor()
{
	if (sliders) {
		delete[] sliders;
		sliders = NULL;
	}
	if (labels) {
		delete[] labels;
		labels = NULL;
	}
}

void ParameterEditor::resized()
{
	int sliderSize = 36;
	int sliderPadding = 40;
	int labelHeight = 16;
	for (int i = 0; i < nSliders; ++i) {
		sliders[i].setBounds(sliderPadding*i, 0, sliderSize, sliderSize);
		labels[i].setBounds(sliderPadding*i, sliderSize, sliderSize, labelHeight);
	}
	setSize(sliderPadding*nSliders, sliderSize + labelHeight);
}

void ParameterEditor::sliderValueChanged(Slider *slider) {
	float value = slider->getValue();
	for (int i = 0; i < nSliders; ++i) {
		if (&sliders[i] == slider) {
			onParameterChanged(i, value);
			editor->parametersChanged();
		}
	}
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