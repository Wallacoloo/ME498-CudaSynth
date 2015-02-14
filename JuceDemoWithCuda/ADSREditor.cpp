#include "ADSREditor.h"

static const char* labelNames[] = { "Start", "A", "Peak", "D", "S", "R", "End", "Stretch" };

// For the case where ADSR envelope starts at 0, moves up to 1, decays to sustain, and then releases to 0
static const float classicAdsrParameterBounds[][2]            = { { 0, 1 },  { 0, 2 }, { 0, 1 },  { 0, 2 }, { 0, 1 },  { 0, 2 }, { 0, 1 }, { -0.8f, 5.0f } };
static const float classicAdsrParameterBoundsPlusOrMinus[][2] = { { -1, 1 }, { 0, 2 }, { -1, 1 }, { 0, 2 }, { -1, 1 }, { 0, 2 }, { -1, 1 }, { -0.8f, 5.0f } };
static const int classicAdsrUsableIndices[]         = { 1, 3, 4, 5, 7, -1 };
// For the case where we have no decay phase and peaks are controllable.
// ADSR starts at arbitrary value, decays to sustain, and then releases to arbitrary value
static const float lfoFreqParameterBounds[][2] = { { 0, 100 }, { 0, 2 }, { 0, 100 }, { 0, 2 }, { 0, 100 }, { 0, 2 }, { 0, 100 }, { -0.8f, 5.0f } };
static const int asrWithPeaksUsableIndices[]        = { 2, 3, 4, 5, 6, 7, -1 };

static const int* usableIndicesFromOptions(ADSREditor::KnobTypes opt) {
	if (opt == ADSREditor::ClassicKnobs) {
		return classicAdsrUsableIndices;
	} else if (opt == ADSREditor::AsrWithPeaksKnobs) {
		return asrWithPeaksUsableIndices;
	}
	return NULL;
}
static const float (*parameterBoundsFromOptions(ADSREditor::KnobLimits opt))[2] {
	if (opt == ADSREditor::NormalizedDepthLimits) {
		return classicAdsrParameterBounds;
	} else if (opt == ADSREditor::NormalizedDepthLimitsPlusOrMinus) {
		return classicAdsrParameterBoundsPlusOrMinus;
	} else if (opt == ADSREditor::LFOFrequencyLimits) {
		return lfoFreqParameterBounds;
	}
	return NULL;
}

ADSREditor::ADSREditor(PluginEditor *editor, ADSR *adsr, const char* editorLabel, KnobTypes knobTypes, KnobLimits knobLimits)
	: ParameterEditor(editor, editorLabel, labelNames, parameterBoundsFromOptions(knobLimits), usableIndicesFromOptions(knobTypes)), adsr(adsr) {
}


ADSREditor::~ADSREditor()
{
}

void ADSREditor::onParameterChanged(int parameterNum, float value) {
	switch (parameterNum) {
	case 0: // Start level
		adsr->setStartLevel(value);
		break;
	case 1: // attack time
		adsr->setAttack(value);
		break;
	case 2: // peak level
		adsr->setPeakLevel(value);
		break;
	case 3: // decay time
		adsr->setDecay(value);
		break;
	case 4: // sustain level
		adsr->setSustain(value);
		break;
	case 5: // release time
		adsr->setRelease(value);
		break;
	case 6: // end level
		adsr->setReleaseLevel(value);
		break;
	case 7: // stretch by partial index
		adsr->setScaleByPartialIdx(value);
		break;
	default:
		break;
	}
}