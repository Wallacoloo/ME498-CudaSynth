// number of audio channels to use (2=stereo)
// This macro serves to avoid placing magic numbers in our code - it is assumed this will always be 2.
#define NUM_CH 2
// Number of partials to include in the sound.
#define NUM_PARTIALS 128

// The maximum number of notes that can be played simultaneously.
#define MAX_SIMULTANEOUS_SYNTH_NOTES 4

//number of samples to buffer at a time.
//larger numbers means fewer transfefs between CPU / GPU,
//  but larger latency
#define BUFFER_BLOCK_SIZE 512
// The delay effect has to calculate its output N samples AHEAD of the current index.
// If we want a maximum of 10sec delay (say 5 voices spaced 2 seconds apart), then we need 10*SAMPLE_RATE buffer size.
// Note: this MUST be a multiple of BUFFER_BLOCK_SIZE
// For reference, 512*512 = 5.9 sec, 512*1024 = 11.9 sec
#define MAX_DELAY_EFFECT_LENGTH (512*BUFFER_BLOCK_SIZE)

// # of audio frames per second
#define SAMPLE_RATE 44100
#define INV_SAMPLE_RATE (1.f/44100.f)