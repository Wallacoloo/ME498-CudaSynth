#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
PluginEditor::PluginEditor(PluginProcessor& owner)
    : AudioProcessorEditor (owner),
      midiKeyboard (owner.keyboardState, MidiKeyboardComponent::horizontalKeyboard),
	  partialLevelsComponent(this, parameterStates.partialLevels),
	  volumeADSR(this, parameterStates.volumeEnvelope.getAdsr(), "Volume", ADSREditor::ClassicKnobs, ADSREditor::NormalizedDepthLimits),
	  volumeLFOFreq(this, parameterStates.volumeEnvelope.getLfo()->getFreqAdsr(), "LFO Freq", ADSREditor::AsrWithPeaksKnobs, ADSREditor::LFOFrequencyLimits),
	  volumeLFODepth(this, parameterStates.volumeEnvelope.getLfo()->getDepthAdsr(), "LFO Depth", ADSREditor::AsrWithPeaksKnobs, ADSREditor::NormalizedDepthLimits),
	  detuneADSR(this, parameterStates.detuneEnvelope.getAdsrLfo()->getAdsr(), "Detune", ADSREditor::ClassicKnobs, ADSREditor::NormalizedDepthLimits),
	  detuneLFOFreq(this, parameterStates.detuneEnvelope.getAdsrLfo()->getLfo()->getFreqAdsr(), "LFO Freq", ADSREditor::AsrWithPeaksKnobs, ADSREditor::LFOFrequencyLimits),
	  detuneLFODepth(this, parameterStates.detuneEnvelope.getAdsrLfo()->getLfo()->getDepthAdsr(), "LFO Depth", ADSREditor::AsrWithPeaksKnobs, ADSREditor::NormalizedDepthLimits)
{
	// add the parameter editors
	addAndMakeVisible(partialLevelsComponent);
	addAndMakeVisible(volumeADSR);
	addAndMakeVisible(volumeLFOFreq);
	addAndMakeVisible(volumeLFODepth);
	addAndMakeVisible(detuneADSR);
	addAndMakeVisible(detuneLFOFreq);
	addAndMakeVisible(detuneLFODepth);

    // add the midi keyboard component
    addAndMakeVisible (midiKeyboard);

    // add the triangular resizer component for the bottom-right of the UI
    addAndMakeVisible (resizer = new ResizableCornerComponent (this, &resizeLimits));
    resizeLimits.setSizeLimits (150, 150, 800, 300);

    // set our component's initial size to be the last one that was stored in the filter's settings
    setSize (owner.lastUIWidth,
             owner.lastUIHeight);

	// initialize synth parameters on device side
	parametersChanged();
}

PluginEditor::~PluginEditor()
{
}

void PluginEditor::paint (Graphics& g)
{
	// paint a solid background
	g.setColour(Colour(0x30, 0x30, 0x30));
    g.fillAll();
}

void PluginEditor::resized()
{
    const int keyboardHeight = 70;

	partialLevelsComponent.setBounds(partialLevelsComponent.getLocalBounds().translated(4, 0));

	int padding = 6;
	ADSREditor* volumeEditors[] = { &volumeADSR, &volumeLFOFreq, &volumeLFODepth };
	ADSREditor* detuneEditors[] = { &detuneADSR, &detuneLFOFreq, &detuneLFODepth };
	Point<int> editorStartPoints[] = { Point<int>(144, 30), Point<int>(4, 120) };
	ADSREditor** editors[] = { volumeEditors, detuneEditors };
	for (int row = 0; row < 2; ++row) {
		Point<int> curPos = editorStartPoints[row];
		for (int i = 0; i < 3; ++i) {
			ADSREditor *cur = editors[row][i];
			Rectangle<int> curBounds = cur->getLocalBounds();
			curBounds.setPosition(curPos);
			cur->setBounds(curBounds);
			curPos = cur->getBounds().getTopRight().translated(padding, 0);
		}
	}

	midiKeyboard.setBounds(4, getHeight() - keyboardHeight - 4, getWidth() - 8, keyboardHeight);

    resizer->setBounds (getWidth() - 16, getHeight() - 16, 16, 16);

    getProcessor().lastUIWidth = getWidth();
    getProcessor().lastUIHeight = getHeight();
}

void PluginEditor::parametersChanged() {
	kernel::parameterStatesChanged(&parameterStates);
}