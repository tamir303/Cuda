#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <math.h>
#include "myProto.h"

    /**
     * Unify all partial histograms
     * 
     * @param h The unified histogram
     * @param temp The partial histograms
     * @param numElements Number of given elements in the data array
     * @param hist_per_thread Number of histograms per thread
     * 
     * @return Unifed histogram array
    */
  __global__  void buildHist(int *h, int *temp, int numElements,  int hist_per_thread) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // current thread index
    int offset_temp = hist_per_thread * RANGE * index; // jumps of 2560

    // For each number and thread, go over all theard's histograms and find current num results
    for (int num = 0; num < RANGE; num++) {
      for (int hist_offset = 0; hist_offset < hist_per_thread; hist_offset++) {
        atomicAdd(&h[num], temp[offset_temp + (hist_offset * RANGE) + num]);
      } 
    }
  }

      /**
     * Build all partial histograms
     * 
     * @param A The data array
     * @param temp The partial histograms
     * @param numElements Number of given elements in the data array
     * @param hist_per_thread Number of histograms per thread
     * 
     * @return Partial histogram array
    */
  __global__  void buildTemp(int *A, int *temp, int numElements, int hist_per_thread) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // current thread index
    int offset_A = (numElements / (NUM_BLOCKS * NUM_THREADS)) * index; // jumps of 2500
    int offset_temp = hist_per_thread * RANGE * index; // jumps of 2560

    // Calculate each inner thread's histogram on chunk of 2500 numbers
    for (int i = 0; i < numElements / (NUM_BLOCKS * NUM_THREADS); i++) {
      temp[offset_temp + A[offset_A + i]]++;
    }
  }

    /**
     * @param h Initiated unified histogram  array
     * 
     * @return historam array initiated with zeros
    */
  __global__  void initHist(int * h) {
    int index = threadIdx.x;
    h[index] = 0;
  }

  /**
   * Initiate all partial histograms
   * 
   * @param temp Initiated partial histograms
   * @param numElements Number of given elements in the data array
   * @param hist_per_thread Number of histograms per thread
   * 
   * @return Partial histogram initiated with zeros
  */
  __global__  void initTemp(int * temp, int numElements, int hist_per_thread) {
  int index = blockIdx.x * blockDim.x + threadIdx.x; // current thread index
  int offset = hist_per_thread * RANGE * index; // jumps of 2560

  for (int i = 0; i < RANGE * hist_per_thread; i++)
    temp[offset + i] = 0;
  }

int computeOnGPU(int *data, int numElements, int* hist) {

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    int hist_per_thread = (int) ceil((numElements / (NUM_BLOCKS * NUM_THREADS)) / RANGE) + 1;
    // Allocate data on device
    int* d_A = NULL;
    err = cudaMalloc((void **)&d_A, numElements * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d cuda (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate hist on device
    int* d_H = NULL;
    err = cudaMalloc((void **)&d_H, RANGE * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d cuda (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate temp on device
    int* d_temp = NULL;
    err = cudaMalloc((void **)&d_temp, (numElements / (NUM_BLOCKS * NUM_THREADS)) * RANGE * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d cuda (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Copy data to the device
    err = cudaMemcpy(d_A, data, numElements * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      fprintf(stderr, "Error in line %d cuda (error code %s)!\n", __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    // Initialize hist on device
    initHist<<<1, RANGE>>>(d_H); // 1 block with 256 threads
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d cuda (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Initialize temp on device
    initTemp<<< NUM_BLOCKS, NUM_THREADS >>>(d_temp, numElements, hist_per_thread);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Error in line %d cuda (error code %s)!\n", __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    // Build partial histograms for each thread
    buildTemp<<< NUM_BLOCKS, NUM_THREADS >>>(d_A, d_temp, numElements, hist_per_thread);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Error in line %d cuda (error code %s)!\n", __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();

    // Unify results
    buildHist<<< NUM_BLOCKS, NUM_THREADS >>>(d_H, d_temp, numElements, hist_per_thread);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Error in line %d cuda (error code %s)!\n", __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();

    // Copy the final histogram to the host
    err = cudaMemcpy(hist, d_H, RANGE * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      fprintf(stderr, "Error in line %d cuda (error code %s)!\n", __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    // Free device global memory
    err = cudaFree(d_A);
    if (err != cudaSuccess) {
      fprintf(stderr, "Error in line %d cuda (error code %s)!\n", __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

        // Free device global memory
    err = cudaFree(d_H);
    if (err != cudaSuccess) {
      fprintf(stderr, "Error in line %d cuda (error code %s)!\n", __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    return 0;
}

