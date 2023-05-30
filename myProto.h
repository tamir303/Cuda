#pragma once

#define SIZE  1000000
#define RANGE  256
#define NUM_THREADS  20
#define NUM_BLOCKS  10

void test(int *data, int n, int* expected_hist);
int computeOnGPU(int *data, int numElements, int* hist);
