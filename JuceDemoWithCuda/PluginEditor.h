/*
  ==============================================================================

    This file was auto-generated by the Jucer!

    It contains the basic startup code for a Juce application.

  ==============================================================================
*/

#ifndef __PLUGINEDITOR_H_4ACCBAA__
#define __PLUGINEDITOR_H_4ACCBAA__

#include "../JuceLibraryCode/JuceHeader.h"
#include "PluginProcessor.h"
#include "PartialLevelsComponent.h"
#include "ADSRComponent.h"
#include "kernel.h"


//==============================================================================
/** This is the editor component that our filter will display.
*/
class JuceDemoPluginAudioProcessorEditor  : public AudioProcessorEditor
{
public:
    JuceDemoPluginAudioProcessorEditor (JuceDemoPluginAudioProcessor&);
    ~JuceDemoPluginAudioProcessorEditor();

    //==============================================================================
    void paint (Graphics&) override;
    void resized() override;

	// Called by other UI components (or self) when the parameters have been manually changed
	void parametersChanged();

private:
	ParameterStates parameterStates;
    MidiKeyboardComponent midiKeyboard;
	PartialLevelsComponent partialLevelsComponent;
	ADSRComponent volumeADSR;
    ScopedPointer<ResizableCornerComponent> resizer;
    ComponentBoundsConstrainer resizeLimits;

    JuceDemoPluginAudioProcessor& getProcessor() const
    {
        return static_cast<JuceDemoPluginAudioProcessor&> (processor);
    }
};


#endif  // __PLUGINEDITOR_H_4ACCBAA__
