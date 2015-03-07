#include "ADSREditor.h"

static const char* labelNames[] = { "Start", "A", "Peak", "D", "S", "R", "End", "Stretch", "Scale" };
static const char* tooltips[] = { "Start Level", "Attack Length", "Peak Level",
"Delay Length", "Sustain Level", "Release Length",
"End Level", "Elongate envelope based on sinusoid's frequency",
"Multiply envelope based on sinusoid's frequency"};

// For the case where ADSR envelope starts at 0, moves up to 1, decays to sustain, and then releases to 0
static const float classicAdsrParameterBounds[][2] = { { 0, 1 }, { 0, 2 }, { 0, 1 }, { 0, 2 }, { 0, 1 }, { 0, 2 }, { 0, 1 }, { -0.8f, 5.0f }, { -1.0f, 5.0f } };
static const float classicAdsrParameterBoundsPlusOrMinus[][2] = { { -1, 1 }, { 0, 2 }, { -1, 1 }, { 0, 2 }, { -1, 1 }, { 0, 2 }, { -1, 1 }, { -0.8f, 5.0f }, { -1.0f, 5.0f } };
static const int classicAdsrUsableIndices[]         = { 1, 3, 4, 5, 7, -1 };
static const int classAdsrNoShiftUsableIndices[] = { 1, 3, 4, 5, -1 };
static const int classApdsrNoShiftUsableIndices[] = { 1, 2, 3, 4, 5, -1 };
static const int classicAdsrWithScaleByIdxUsableIndices[] = { 1, 3, 4, 5, 7, 8, -1 };
// For the case where we have no decay phase and peaks are controllable.
// ADSR starts at arbitrary value, decays to sustain, and then releases to arbitrary value
static const float lfoFreqParameterBounds[][2] = { { 0, 100 }, { 0, 2 }, { 0, 100 }, { 0, 2 }, { 0, 100 }, { 0, 2 }, { 0, 100 }, { -0.8f, 5.0f }, { -1.0f, 5.0f } };
static const int asrWithPeaksUsableIndices[]        = { 2, 3, 4, 5, 6, 7, 8, -1 };
// For the filter shift envelope:
static const float classicAdsrFilterParameterBounds[][2] = { { -NYQUIST_RATE_RAD, NYQUIST_RATE_RAD }, { 0, 2 }, { -NYQUIST_RATE_RAD, NYQUIST_RATE_RAD }, { 0, 2 }, { -NYQUIST_RATE_RAD, NYQUIST_RATE_RAD }, { 0, 2 }, { -NYQUIST_RATE_RAD, NYQUIST_RATE_RAD }, { -0.8f, 5.0f }, { -1.0f, 5.0f } };

static const int* usableIndicesFromOptions(ADSREditor::KnobTypes opt) {
	if (opt == ADSREditor::ClassicKnobs) {
		return classicAdsrUsableIndices;
	} else if (opt == ADSREditor::ClassicKnobsNoShiftByIdx) {
		return classAdsrNoShiftUsableIndices;
	} else if (opt == ADSREditor::ClassicKnobsWithPeakNoShiftByIdx) {
		return classApdsrNoShiftUsableIndices;
	} else if (opt == ADSREditor::ClassicKnobsWithScaleByIdx) {
		return classicAdsrWithScaleByIdxUsableIndices;
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
	} else if (opt == ADSREditor::FreqFilterDepthLimits) {
		return classicAdsrFilterParameterBounds;
	} else if (opt == ADSREditor::LFOFrequencyLimits) {
		return lfoFreqParameterBounds;
	}
	return NULL;
}

ADSREditor::ADSREditor(PluginEditor *editor, ADSR *adsr, const char* editorLabel, KnobTypes knobTypes, KnobLimits knobLimits)
	: ParameterEditor(editor, editorLabel, labelNames, tooltips, parameterBoundsFromOptions(knobLimits), usableIndicesFromOptions(knobTypes)), 
	adsr(adsr) {
	// sync GUI with parameter values
	refreshSliderValues();
}


ADSREditor::~ADSREditor()
{
}

float ADSREditor::getParameterValue(int parameterNum) const {
	switch (parameterNum) {
	case 0:
		return adsr->getStartLevel();
	case 1:
		return adsr->getAttack();
	case 2:
		return adsr->getPeakLevel();
	case 3:
		return adsr->getDecay();
	case 4:
		return adsr->getSustain();
	case 5: 
		return adsr->getRelease();
	case 6:
		return adsr->getReleaseLevel();
	case 7:
		return adsr->getScaleByPartialIdx();
	case 8:
		return adsr->getAmplificationByPartialIdx();
	default:
		return 0.f;
	}
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
	case 8:
		adsr->setAmplificationByPartialIdx(value);
		break;
	default:
		break;
	}
}