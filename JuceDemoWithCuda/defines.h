//Can either use CPU processing or Cuda processing
#ifndef USE_CUDA
	#define USE_CUDA 0
#endif

//number of samples to buffer at a time.
//larger numbers means fewer transfefs between CPU / GPU,
//  but larger latency
#define BUFFER_BLOCK_SIZE 512