#ifndef PARTIALLEVELSCOMPONENT_H
#define PARTIALLEVELSCOMPONENT_H

#include "../JuceLibraryCode/JuceHeader.h"

class PluginEditor;

class PartialLevelsComponent : public Component {
	PluginEditor *editor;
	float *partialLevels;
public:
	PartialLevelsComponent(PluginEditor *editor, float *partialLevels);
	~PartialLevelsComponent();
	void paint(Graphics& g) override;
	void mouseDrag(const MouseEvent &event) override;
};

#endif
