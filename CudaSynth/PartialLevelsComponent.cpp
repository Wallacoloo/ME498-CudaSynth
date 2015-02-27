#include "PartialLevelsComponent.h"
#include "defines.h"
#include "PluginEditor.h"

#include <algorithm>

#define PARTIAL_EDITOR_WIDTH 128
#define PARTIAL_EDITOR_HEIGHT 100


PartialLevelsComponent::PartialLevelsComponent(PluginEditor *editor, float *partialLevels)
	: editor(editor),
	  partialLevels(partialLevels)
{
	setSize(PARTIAL_EDITOR_WIDTH, PARTIAL_EDITOR_HEIGHT);
}


PartialLevelsComponent::~PartialLevelsComponent()
{
}

void PartialLevelsComponent::paint(Graphics &g)
{
	//g.fillAll(Colours::black);
	g.setColour(Colour(0xBD, 0x00, 0x00));
	for (int p = 0; p < NUM_PARTIALS; ++p) {
		int pxPerPartial = getWidth() / NUM_PARTIALS;
		int x = pxPerPartial * p;
		int w = std::min(3, pxPerPartial - 1);
		int top = getHeight() * (1.0 - partialLevels[p]);
		//g.drawVerticalLine(x, top, getHeight());
		g.fillRect(x, top, w, getHeight());
	}
}

void PartialLevelsComponent::updateFromMouseEvent(const MouseEvent &event) {
	int x = event.getPosition().getX();
	int pxPerPartial = getWidth() / NUM_PARTIALS;
	int partialIdx = (x+0.5*pxPerPartial) / pxPerPartial;
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