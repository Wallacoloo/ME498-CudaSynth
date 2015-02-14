#include "ParameterEditor.h"
#include "PluginEditor.h"

ParameterEditor::ParameterEditor(PluginEditor *editor, const char** labelNames, const float parameterBounds[][2], const int* usableParameterIndices)
	: editor(editor), parameterIndexMap(usableParameterIndices)
{
	// determine the number of sliders
	for (nSliders = 0; usableParameterIndices[nSliders] != -1; ++nSliders) {}
	sliders = new Slider[nSliders];
	labels = new Label[nSliders];
	
	// initialize relevant sliders & labels
	for (int usableParamsIdx = 0; usableParamsIdx < nSliders; ++usableParamsIdx) {
		int i = usableParameterIndices[usableParamsIdx];
		Slider *slider = &sliders[usableParamsIdx];
		Label *label = &labels[usableParamsIdx];
		slider->setSliderStyle(Slider::Rotary);
		slider->setTextBoxStyle(Slider::NoTextBox, false, 0, 0);
		addAndMakeVisible(slider);
		slider->addListener(this);
		slider->setRange(parameterBounds[i][0], parameterBounds[i][1], 0.f);
		addAndMakeVisible(label);
		label->setFont(Font(11.f));
		label->setText(labelNames[i], sendNotification);
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
			onParameterChanged(parameterIndexMap[i], value);
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