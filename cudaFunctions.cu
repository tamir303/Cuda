#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"

  __global__  void buildHist(int *h, int *array, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (tid < size)
      atomicAdd(&h[array[tid]], 1);
      tid += stride;
  }

  __global__  void initHist(int * h) {
    int index = threadIdx.x;
    h[index] = 0;
  }

int computeOnGPU(int *data, int numElements, int* hist) {

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

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

    // Unify the results
    buildHist<<< NUM_BLOCKS, NUM_THREADS >>>(d_H, d_A, numElements);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Error in line %d cuda (error code %s)!\n", __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

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

