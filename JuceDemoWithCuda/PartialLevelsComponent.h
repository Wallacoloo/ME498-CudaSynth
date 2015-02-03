#ifndef PARTIALLEVELSCOMPONENT_H
#define PARTIALLEVELSCOMPONENT_H

#include "../JuceLibraryCode/JuceHeader.h"

class JuceDemoPluginAudioProcessorEditor;

class PartialLevelsComponent : public Component {
	JuceDemoPluginAudioProcessorEditor *editor;
	float *partialLevels;
public:
	PartialLevelsComponent(JuceDemoPluginAudioProcessorEditor *editor, float *partialLevels);
	~PartialLevelsComponent();
	void paint(Graphics& g) override;
	void mouseDrag(const MouseEvent &event) override;
};

#endif
