#ifndef PIECEWISEEDITOR_H
#define PIECEWISEEDITOR_H

#include "../JuceLibraryCode/JuceHeader.h"
#include "kernel.h"

class PluginEditor;

class PiecewiseEditor : public Component
{
	PluginEditor *editor;
	PiecewiseFunction *func;
	int idxOfDraggingPoint;


	float pxPerUnitX() const;
	float pxPerUnitY() const;
	float clampFreq(float f) const;
	float clampGain(float g) const;
	void updateFromMouseEvent(const MouseEvent &event);
public:
	PiecewiseEditor(PluginEditor *editor, PiecewiseFunction *func);
	~PiecewiseEditor();
	void paint(Graphics& g) override;
	void mouseDown(const MouseEvent &event) override;
	void mouseDrag(const MouseEvent &event) override;
};

#endif