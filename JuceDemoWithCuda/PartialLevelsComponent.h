#ifndef PARTIALLEVELSCOMPONENT_H
#define PARTIALLEVELSCOMPONENT_H

#include "../JuceLibraryCode/JuceHeader.h"

class PartialLevelsComponent :
	public Component
{
	float *partialLevels;
public:
	PartialLevelsComponent(float *partialLevels);
	~PartialLevelsComponent();
	void paint(Graphics& g) override;
	void mouseDrag(const MouseEvent &event) override;
};

#endif
