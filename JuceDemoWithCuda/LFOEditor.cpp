#include "LFOEditor.h"

static const char* labelNames[] = { "Freq", "Depth" };
static const float parameterBounds[][2] = { { 0, 100 }, { 0, 1 } };
static const int usableParameterIndices[] = { 0, 1 , -1};


LFOEditor::LFOEditor(PluginEditor *editor, LFO *lfo)
	: ParameterEditor(editor, labelNames, parameterBounds, usableParameterIndices), lfo(lfo) {
}


LFOEditor::~LFOEditor()
{
}

void LFOEditor::onParameterChanged(int parameterNum, float value) {
	switch (parameterNum) {
	case 0:
		lfo->setLfoFreq(value);
		break;
	case 1:
		lfo->setLfoDepth(value);
		break;
	default:
		break;
	}
}