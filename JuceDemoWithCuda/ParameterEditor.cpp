#include "ParameterEditor.h"
#include "PluginEditor.h"

ParameterEditor::ParameterEditor(PluginEditor *editor, const char* editorLabelText, const char** labelNames, const float parameterBounds[][2], const int* usableParameterIndices)
	: editor(editor), parameterIndexMap(usableParameterIndices)
{
	// configure the label for this editor
	addAndMakeVisible(editorLabel);
	editorLabel.setFont(Font(14.f, Font::bold));
	editorLabel.setJustificationType(Justification::centred);
	editorLabel.setText(editorLabelText, sendNotification);

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
		label->setJustificationType(Justification::centred);
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
	int sliderLabelHeight = 16;
	int mainLabelHeight = 20;
	int fullWidth = sliderPadding*nSliders;
	int fullHeight = mainLabelHeight + sliderSize + sliderLabelHeight;

	// position the overall editor label
	editorLabel.setBounds(0, 0, fullWidth, mainLabelHeight);

	// position the sliders & their labels
	for (int i = 0; i < nSliders; ++i) {
		sliders[i].setBounds(sliderPadding*i, mainLabelHeight, sliderSize, sliderSize);
		labels[i].setBounds(sliderPadding*i, mainLabelHeight+sliderSize, sliderSize, sliderLabelHeight);
	}
	// set the desired size
	setSize(fullWidth, fullHeight);
}

void ParameterEditor::sliderValueChanged(Slider *slider) {
	float value = slider->getValue();

	// trigger a virtual callback with the slider index and value
	for (int i = 0; i < nSliders; ++i) {
		if (&sliders[i] == slider) {
			onParameterChanged(parameterIndexMap[i], value);
			editor->parametersChanged();
		}
	}
}

void ParameterEditor::paint(Graphics& g)
{
	// fill a background
	g.setColour(Colour(0xA0A0A0).withAlpha(0.5f));
	g.fillAll();

	// fill a rectangular outline
	g.setColour(Colour(0xBF6300).withAlpha(1.0f));
	Rectangle<int> bounds = getLocalBounds();
	g.drawRect(bounds, 2.0f);
}
