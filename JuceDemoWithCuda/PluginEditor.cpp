#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
PluginEditor::PluginEditor(PluginProcessor& owner)
    : AudioProcessorEditor (owner),
      midiKeyboard (owner.keyboardState, MidiKeyboardComponent::horizontalKeyboard),
	  partialLevelsComponent(this, parameterStates.partialLevels),
	  volumeADSR(this, parameterStates.volumeEnvelope.getAdsr()),
	  volumeLFOFreq(this, parameterStates.volumeEnvelope.getLfo()->getFreqAdsr()),
	  volumeLFODepth(this, parameterStates.volumeEnvelope.getLfo()->getDepthAdsr())
{
	// add the parameter editors
	addAndMakeVisible(partialLevelsComponent);
	addAndMakeVisible(volumeADSR);
	addAndMakeVisible(volumeLFOFreq);
	addAndMakeVisible(volumeLFODepth);

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
	// paint a gradient background
    g.setGradientFill (ColourGradient (Colours::white, 0, 0,
                                       Colours::grey, 0, (float) getHeight(), false));
    g.fillAll();
}

void PluginEditor::resized()
{

    const int keyboardHeight = 70;
    midiKeyboard.setBounds (4, getHeight() - keyboardHeight - 4, getWidth() - 8, keyboardHeight);

	// take the volumeADSR's desired size, and offset it by what we want the location to be
	Rectangle<int> volBounds = volumeADSR.getLocalBounds();
	volBounds.setPosition(144, 56);
	volumeADSR.setBounds(volBounds);
	// take volumeLFOFreq's desired size and position it to the right of the ADSR
	Rectangle<int> lfoFreqBounds = volumeLFOFreq.getLocalBounds();
	lfoFreqBounds.setPosition(volBounds.getTopRight());
	volumeLFOFreq.setBounds(lfoFreqBounds);
	// take volumeLFODepths's desired size and position it to the right of the LFOFreq
	Rectangle<int> lfoDepthBounds = volumeLFODepth.getLocalBounds();
	lfoDepthBounds.setPosition(lfoFreqBounds.getTopRight());
	volumeLFODepth.setBounds(lfoDepthBounds);

    resizer->setBounds (getWidth() - 16, getHeight() - 16, 16, 16);

    getProcessor().lastUIWidth = getWidth();
    getProcessor().lastUIHeight = getHeight();
}

void PluginEditor::parametersChanged() {
	kernel::parameterStatesChanged(&parameterStates);
}