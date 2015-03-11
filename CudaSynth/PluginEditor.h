#ifndef __PLUGINEDITOR_H_4ACCBAA__
#define __PLUGINEDITOR_H_4ACCBAA__

#include "JuceLibraryCode/JuceHeader.h"
#include "PluginProcessor.h"
#include "PartialLevelsComponent.h"
#include "ADSREditor.h"
#include "DetuneRandEditor.h"
#include "PiecewiseEditor.h"
#include "kernel.h"

class PluginEditor  : public AudioProcessorEditor
{
public:
    PluginEditor (PluginProcessor&);
    ~PluginEditor();

    //==============================================================================
    void paint (Graphics&) override;
    void resized() override;

	// Called by other UI components (or self) when the parameters have been manually changed
	void parametersChanged();

private:
	TooltipWindow tooltipWindow;
	ParameterStates parameterStates;
    MidiKeyboardComponent midiKeyboard;
	PartialLevelsComponent partialLevelsComponent;
	ADSREditor volumeADSR, volumeLFOFreq, volumeLFODepth;
	ADSREditor stereoADSR, stereoLFOFreq, stereoLFODepth;
	DetuneRandEditor detuneRandEditor;
	ADSREditor detuneADSR, detuneLFOFreq, detuneLFODepth;
	ADSREditor delaySpaceADSR;
	ADSREditor delayAmpLossADSR;
	PiecewiseEditor filterComponent;
	ADSREditor filterADSR;
    ScopedPointer<ResizableCornerComponent> resizer;
    ComponentBoundsConstrainer resizeLimits;

    PluginProcessor& getProcessor() const
    {
		return static_cast<PluginProcessor&> (processor);
    }
};


#endif  // __PLUGINEDITOR_H_4ACCBAA__
