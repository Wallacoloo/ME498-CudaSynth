/*
==============================================================================

JuceDemoPlugin.cpp
Created: 16 Jan 2015 7:54:51am
Author:  Colin

==============================================================================
*/

#include "../JuceLibraryCode/JuceHeader.h"
#include "juce_audio_plugin_client/Standalone/juce_StandaloneFilterWindow.h"

class StandalonePlugin : public JUCEApplication
{
public:
	StandalonePlugin()
	{
	}

	void initialise(const String& commandLineParameters)
	{
		//ApplicationProperties::getInstance()->setStorageParameters(T("JuceDemoPlugin"), String::empty, T("JuceDemoPlugin"), 400, PropertiesFile::storeAsXML);
		filterWindow = new StandaloneFilterWindow(String("Ctrlr v4"), Colours::black, NULL, false);
		filterWindow->setTitleBarButtonsRequired(DocumentWindow::allButtons, false);
		filterWindow->setVisible(true);
		filterWindow->setResizable(true, true);
		filterWindow->getDeviceManager().playTestSound();
	}

	void shutdown()
	{
		deleteAndZero(filterWindow);
	}

	const String getApplicationName()
	{
		return "JuceDemoPlugin";
	}

	const String getApplicationVersion()
	{
		return "4.0";
	}

private:
	StandaloneFilterWindow *filterWindow;
};

//if running standalone, then we must insert the program entry point
#if !JucePlugin_Build_VST
//Should be able to use START_JUCE_APPLICATION macro from juice_events/messages/juce_Initialisation.h,
//but it looks like it's getting empty-defined for some reason, so I copied the macro and inserted it inline:
//START_JUCE_APPLICATION(StandalonePlugin)
static juce::JUCEApplicationBase* juce_CreateApplication() { return new StandalonePlugin(); } 
int main() {
	juce::JUCEApplicationBase::createInstance = &juce_CreateApplication;
	return juce::JUCEApplicationBase::main(JUCE_MAIN_FUNCTION_ARGS);
}
#endif