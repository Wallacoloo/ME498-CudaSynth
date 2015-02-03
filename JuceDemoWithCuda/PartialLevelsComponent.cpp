#include "PartialLevelsComponent.h"
#include "defines.h"
#include "PluginEditor.h"


PartialLevelsComponent::PartialLevelsComponent(JuceDemoPluginAudioProcessorEditor *editor, float *partialLevels)
	: editor(editor),
	  partialLevels(partialLevels)
{
	setSize(NUM_PARTIALS, 100);
	for (int p = 0; p < NUM_PARTIALS; ++p) {
		partialLevels[p] = 0.5;
	}
}


PartialLevelsComponent::~PartialLevelsComponent()
{
}

void PartialLevelsComponent::paint(Graphics &g)
{
	//g.fillAll(Colours::black);
	g.setColour(Colour(0xBD, 0x00, 0x00));
	for (int p = 0; p < NUM_PARTIALS; ++p) {
		int top = getHeight() * (1.0 - partialLevels[p]);
		g.drawVerticalLine(p, top, getHeight());
	}
}


void PartialLevelsComponent::mouseDrag(const MouseEvent &event) {
	printf("MouseDrag: %i, %i, h: %i\n", event.getPosition().getX(), event.getPosition().getY(), getHeight());
	int partialIdx = event.getPosition().getX();
	if (0 <= partialIdx && partialIdx < NUM_PARTIALS) {
		float level = 1.0 - event.getPosition().getY() / (float)getHeight();
		partialLevels[partialIdx] = level;
		editor->parametersChanged();
		repaint();
	}
}