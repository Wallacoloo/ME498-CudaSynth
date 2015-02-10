#include "LFOEditor.h"

static const char* labelNames[] = { "Freq", "Depth" };

LFOEditor::LFOEditor(PluginEditor *editor, LFO *lfo)
	: ParameterEditor(editor, 2, labelNames), lfo(lfo) {
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