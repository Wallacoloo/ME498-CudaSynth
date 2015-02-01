//Can either use CPU processing or Cuda processing
#ifndef USE_CUDA
	#define USE_CUDA 0
#endif

//number of samples to buffer at a time.
//larger numbers means fewer transfefs between CPU / GPU,
//  but larger latency
#define BUFFER_BLOCK_SIZE 512
//number of audio channels to use (2=stereo)
//This macro serves to avoid placing magic numbers in our code - it is assumed this will always be 2.
#define NUM_CH 2
//Number of partials to include in the sound.
#define NUM_PARTIALS 128

#define SAMPLE_RATE 44100
#define INV_SAMPLE_RATE (1.f/44100.f)