// set to 1 to disable CUDA even on cuda-enabled machines
#ifndef NEVER_USE_CUDA
#define NEVER_USE_CUDA 0
#endif

// number of audio channels to use (2=stereo)
// This macro serves to avoid placing magic numbers in our code - it is assumed this will always be 2.
#define NUM_CH 2
// Number of partials to include in the sound.
#define NUM_PARTIALS 128

// The maximum number of notes that can be played simultaneously.
#define MAX_SIMULTANEOUS_SYNTH_NOTES 4

// number of samples to buffer at a time.
// larger numbers means fewer transfefs between CPU / GPU,
//   but larger latency
#define BUFFER_BLOCK_SIZE 512
#define INV_BUFFER_BLOCK_SIZE (1.f / BUFFER_BLOCK_SIZE)
// number of threads to use for evaluating *each* partial within the buffer block.
#define NUM_THREADS_PER_PARTIAL 512
#define NUM_SAMPLES_PER_THREAD (BUFFER_BLOCK_SIZE / NUM_THREADS_PER_PARTIAL)
// The delay effect has to calculate its output N samples AHEAD of the current index.
// If we want a maximum of 10sec delay (say 5 voices spaced 2 seconds apart), then we need 10*SAMPLE_RATE buffer size.
// Note: this MUST be a multiple of BUFFER_BLOCK_SIZE
// For reference, 512*512 = 5.9 sec, 512*1024 = 11.9 sec
#define MAX_DELAY_EFFECT_LENGTH (512*BUFFER_BLOCK_SIZE)

// # of audio frames per second
#define SAMPLE_RATE 44100
#define INV_SAMPLE_RATE (1.f/44100.f)

#define PI 3.14159265358979323846



// Expose conveniences:
#ifdef __CUDACC__
	// allow to declare function prototype as HOST both when compiling in Cuda and plain C++.
	#define HOST __host__
	#define DEVICE __device__
#else
	#define HOST
	#define DEVICE
#endif