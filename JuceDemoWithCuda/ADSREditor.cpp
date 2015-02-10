#include "ADSREditor.h"

static const char* labelNames[] = { "A", "D", "S", "R", "Stretch" };

ADSREditor::ADSREditor(PluginEditor *editor, ADSR *adsr)
	: ParameterEditor(editor, 5, labelNames), adsr(adsr) {
}


ADSREditor::~ADSREditor()
{
}

void ADSREditor::onParameterChanged(int parameterNum, float value) {
	switch (parameterNum) {
	case 0:
		adsr->setAttack(value);
		break;
	case 1:
		adsr->setDecay(value);
		break;
	case 2:
		adsr->setSustain(value);
		break;
	case 3:
		adsr->setRelease(value);
		break;
	case 4:
		adsr->setScaleByPartialIdx(value);
		break;
	default:
		break;
	}
}