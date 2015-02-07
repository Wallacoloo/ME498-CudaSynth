#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
JuceDemoPluginAudioProcessorEditor::JuceDemoPluginAudioProcessorEditor (JuceDemoPluginAudioProcessor& owner)
    : AudioProcessorEditor (owner),
      midiKeyboard (owner.keyboardState, MidiKeyboardComponent::horizontalKeyboard),
	  partialLevelsComponent(this, parameterStates.partialLevels)
{
	addAndMakeVisible(partialLevelsComponent);

	addAndMakeVisible(volumeADSR);

    // add the midi keyboard component..
    addAndMakeVisible (midiKeyboard);

    // add the triangular resizer component for the bottom-right of the UI
    addAndMakeVisible (resizer = new ResizableCornerComponent (this, &resizeLimits));
    resizeLimits.setSizeLimits (150, 150, 800, 300);

    // set our component's initial size to be the last one that was stored in the filter's settings
    setSize (owner.lastUIWidth,
             owner.lastUIHeight);

	parametersChanged();
}

JuceDemoPluginAudioProcessorEditor::~JuceDemoPluginAudioProcessorEditor()
{
}

//==============================================================================
void JuceDemoPluginAudioProcessorEditor::paint (Graphics& g)
{
    g.setGradientFill (ColourGradient (Colours::white, 0, 0,
                                       Colours::grey, 0, (float) getHeight(), false));
    g.fillAll();
}

void JuceDemoPluginAudioProcessorEditor::resized()
{

    const int keyboardHeight = 70;
    midiKeyboard.setBounds (4, getHeight() - keyboardHeight - 4, getWidth() - 8, keyboardHeight);

	volumeADSR.setBounds(144, 68, 192, 100);
	//attackSlider.setBounds(256, 60, 150, 40);

    resizer->setBounds (getWidth() - 16, getHeight() - 16, 16, 16);

    getProcessor().lastUIWidth = getWidth();
    getProcessor().lastUIHeight = getHeight();
}

// This is our Slider::Listener callback, when the user drags a slider.
/*void JuceDemoPluginAudioProcessorEditor::sliderValueChanged (Slider* slider)
{
    if (slider == &gainSlider)
    {
        // It's vital to use setParameterNotifyingHost to change any parameters that are automatable
        // by the host, rather than just modifying them directly, otherwise the host won't know
        // that they've changed.
        getProcessor().setParameterNotifyingHost (JuceDemoPluginAudioProcessor::gainParam,
                                                  (float) gainSlider.getValue());
    }
    else if (slider == &delaySlider)
    {
        getProcessor().setParameterNotifyingHost (JuceDemoPluginAudioProcessor::delayParam,
                                                  (float) delaySlider.getValue());
    }
}*/

void JuceDemoPluginAudioProcessorEditor::parametersChanged() {
	kernel::parameterStatesChanged(&parameterStates);
}