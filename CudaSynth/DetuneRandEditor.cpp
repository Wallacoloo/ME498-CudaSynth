#include "DetuneRandEditor.h"

static const char* labelNames[] = { "seed", "mix" };
static const char* tooltips[] = { "Random Seed", "Mix level of randomness" };
static const float detuneRandParameterBounds[][2] = { { 0, 1 }, { 0, 1 } };
static const int detuneRandUsableIndices[] = { 0, 1, -1 };


DetuneRandEditor::DetuneRandEditor(PluginEditor *editor, DetuneEnvelope *detune, const char* editorLabel)
	: ParameterEditor(editor, editorLabel, labelNames, tooltips, detuneRandParameterBounds, detuneRandUsableIndices),
	detune(detune)
{
}


DetuneRandEditor::~DetuneRandEditor()
{
}

float DetuneRandEditor::getParameterValue(int parameterNum) const {
	switch (parameterNum) {
	case 0:
		return detune->getRandSeed();
	case 1:
		return detune->getRandMix();
	default:
		return 0.f;
	}
}

void DetuneRandEditor::onParameterChanged(int parameterNum, float value) {
	switch (parameterNum) {
	case 0: // Seed
		detune->setRandSeed(value);
		break;
	case 1: // Mix
		detune->setRandMix(value);
		break;
	default:
		break;
	}
}