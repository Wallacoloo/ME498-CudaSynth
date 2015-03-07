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
	return getWidth() / NYQUIST_RATE_RAD;
}
float PiecewiseEditor::pxPerUnitY() const {
	return getHeight() / 10.f;
}
float PiecewiseEditor::clampFreq(float f) const {
	return std::max(0.f, std::min((float)NYQUIST_RATE_RAD, f));
}
float PiecewiseEditor::clampGain(float g) const {
	return std::min(10.f, std::max(0.f, g));
}

void PiecewiseEditor::paint(Graphics &g)
{
	//g.fillAll(Colours::black);
	g.setColour(Colour(0xBD, 0x00, 0x00));
	g.setFillType(FillType(Colour(0xBD, 0x00, 0x00)));
	float xScale = pxPerUnitX();
	float yScale = pxPerUnitY();
	float height = getHeight();
	float ptRad = 2.5f;
	float x = 0.f;
	float y = 0.f;
	int numPoints = func->numPoints();
	for (int p = 0; p < numPoints; ++p) {
		float newX = func->startTimeOfPiece(p)*xScale;
		float newY = height-func->startLevelOfPiece(p)*yScale;
		// show the line
		if (p != 0) {
			g.drawLine(x, y, newX, newY);
		}
		// draw the endpoint
		g.fillEllipse(newX-ptRad, newY-ptRad, ptRad*2+1, ptRad*2+1);
		x = newX;
		y = newY;
	}
}

void PiecewiseEditor::updateFromMouseEvent(const MouseEvent &event) {
	// exit if not dragging a point
	if (idxOfDraggingPoint < 0) {
		return;
	}
	float height = getHeight();
	int x = event.getPosition().getX();
	int y = height-event.getPosition().getY();
	float t = clampFreq(x/pxPerUnitX());
	float v = clampGain(y / pxPerUnitY());
	func->movePoint(idxOfDraggingPoint, t, v);
	editor->parametersChanged();
	repaint();
}

void PiecewiseEditor::mouseDown(const MouseEvent &event) {
	// determine the index of the point that was clicked.
	// first, null the dragging index.
	idxOfDraggingPoint = -1;
	MouseEvent relEvt = event.getEventRelativeTo(this);
	float mx = relEvt.getMouseDownX();
	float my = relEvt.getMouseDownY();
	float xScale = pxPerUnitX();
	float yScale = pxPerUnitY();
	float height = getHeight();
	
	if (relEvt.getNumberOfClicks() == 1) {
		// if single-click, then determine the point to drag
		float thresh = 81.f; // max distance^2 to grab a point
		int numPoints = func->numPoints();
		for (int p = 0; p < numPoints; ++p) {
			float x = func->startTimeOfPiece(p)*xScale;
			float y = height - func->startLevelOfPiece(p)*yScale;
			if ((x - mx)*(x - mx) + (y - my)*(y - my) <= thresh) {
				idxOfDraggingPoint = p;
			}
		}
	} else if (relEvt.getNumberOfClicks() == 2) {
		// double click: insert a new point
		float t = clampFreq(mx / pxPerUnitX());
		float v = clampGain((height - my) / pxPerUnitY());
		idxOfDraggingPoint = func->insertPoint(t, v);
	}
	if (relEvt.mods.isLeftButtonDown() && !relEvt.mods.isCtrlDown()) {
		// move point if it was a left-click
		updateFromMouseEvent(relEvt);
	} else if (relEvt.mods.isLeftButtonDown() && relEvt.mods.isCtrlDown()) {
		// delete point if ctrl + left-click;
		if (idxOfDraggingPoint != -1) {
			func->removePoint(idxOfDraggingPoint);
			idxOfDraggingPoint = -1;
			editor->parametersChanged();
			repaint();
		}
	}
}


void PiecewiseEditor::mouseDrag(const MouseEvent &event) {
	if (event.mods.isLeftButtonDown() && !event.mods.isCtrlDown()) {
		updateFromMouseEvent(event.getEventRelativeTo(this));
	}
}