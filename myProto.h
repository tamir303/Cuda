#pragma once

#define SIZE  1000000
#define RANGE  256
#define NUM_THREADS  20
#define NUM_BLOCKS  10

void test(int *data, int n, int* expected_hist);
int computeOnGPU(int *data, int numElements, int* hist);
void combine_hist(int* h_1, int* h_2);
void random_array(int* arr);
void print_arr(int* arr);