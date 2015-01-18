//Can either use CPU processing or Cuda processing
#define USE_CUDA 0

//number of samples to buffer at a time.
//larger numbers means fewer transfefs between CPU / GPU,
//  but larger latency
#define BUFFER_BLOCK_SIZE 512