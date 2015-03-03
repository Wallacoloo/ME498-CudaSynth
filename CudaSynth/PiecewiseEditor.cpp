#include "PiecewiseEditor.h"

#include "PluginEditor.h"


PiecewiseEditor::PiecewiseEditor(PluginEditor *editor, PiecewiseFunction *func)
	: editor(editor), func(func), idxOfDraggingPoint(-1)
{
}


PiecewiseEditor::~PiecewiseEditor()
{
}

float PiecewiseEditor::pxPerUnitX() const {
	return 0.1;
}
float PiecewiseEditor::pxPerUnitY() const {
	return 20;
}

void PiecewiseEditor::paint(Graphics &g)
{
	//g.fillAll(Colours::black);
	g.setColour(Colour(0xBD, 0x00, 0x00));
	g.setFillType(FillType(Colour(0xBD, 0x00, 0x00)));
	float xScale = pxPerUnitX();
	float yScale = pxPerUnitY();
	float x = 0.f;
	float y = 0.f;
	for (int p = 0; p < func->numPoints(); ++p) {
		float newX = func->startTimeOfPiece(p)*xScale;
		float newY = func->startLevelOfPiece(p)*yScale;
		// show the line
		if (p != 0) {
			g.drawLine(x, y, newX, newY);
		}
		// draw the endpoint
		g.fillEllipse(newX, newY, 3, 3);
		x = newX;
		y = newY;
	}
}

void PiecewiseEditor::updateFromMouseEvent(const MouseEvent &event) {
	// exit if not dragging a point
	if (idxOfDraggingPoint < 0) {
		return;
	}
	int x = event.getPosition().getX();
	int y = event.getPosition().getY();
	float t = x/pxPerUnitX();
	float v = y/pxPerUnitY();
	func->movePoint(idxOfDraggingPoint, t, v);
	repaint();
}

void PiecewiseEditor::mouseDown(const MouseEvent &event) {
	updateFromMouseEvent(event);
}


void PiecewiseEditor::mouseDrag(const MouseEvent &event) {
	updateFromMouseEvent(event);
}