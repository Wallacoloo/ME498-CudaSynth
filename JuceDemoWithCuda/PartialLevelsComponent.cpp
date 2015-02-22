#include "PartialLevelsComponent.h"
#include "defines.h"
#include "PluginEditor.h"

#include <algorithm>


PartialLevelsComponent::PartialLevelsComponent(PluginEditor *editor, float *partialLevels)
	: editor(editor),
	  partialLevels(partialLevels)
{
	setSize(NUM_PARTIALS, 100);
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

void PartialLevelsComponent::updateFromMouseEvent(const MouseEvent &event) {
	int partialIdx = event.getPosition().getX();
	if (0 <= partialIdx && partialIdx < NUM_PARTIALS) {
		float level = 1.0f - event.getPosition().getY() / (float)getHeight();
		// clamp the level to between 0 and 1.
		level = std::max(0.0f, std::min(1.0f, level));
		partialLevels[partialIdx] = level;
		editor->parametersChanged();
		repaint();
	}
}

void PartialLevelsComponent::mouseDown(const MouseEvent &event) {
	updateFromMouseEvent(event);
}


void PartialLevelsComponent::mouseDrag(const MouseEvent &event) {
	updateFromMouseEvent(event);
}