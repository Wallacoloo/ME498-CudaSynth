# ME498-CudaSynth
VST synthesizer that offloads processing to Cuda-enabled devices

Demo
=======
[Demo Video 1](https://www.youtube.com/watch?v=xUN_3zn8Ivk) (3:01)

Prerequisites
========
To build this you will need the Steinberg VST SDK.
You can follow the instructions on their website [here](http://www.steinberg.net/en/company/developers.html)
Then download the "VST Audio Plug-Ins SDK (Version 3.6.0)" package listed on that webpage and copy it to `c:\SDKs\VST3 SDK` (or another location so long as you update the C++ include paths).

Project files are provided for Microsoft Visual Studio. The code has not been tested on any platforms other than Windows or with any compilers besides MSVC.
