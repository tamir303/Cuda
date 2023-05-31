#include "myProto.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void test(int *data, int n, int* expected_hist) {
    int i;
    int* iterative_hist = (int *)calloc(n, sizeof(int));
    for (i = 0;   i < n;   i++) {
        iterative_hist[data[i]]++;
    }
    for (i = 0; i < RANGE; i++)
        if (iterative_hist[i] != expected_hist[i]) {
            fprintf(stderr, "\nTest fail for number %d, expected %d but got %d\n", i, expected_hist[i], iterative_hist[i]);
            exit(1);
        }
    printf("\nThe test passed successfully\n"); 
}

void combine_hist(int* h_1, int* h_2) {
   #pragma omp paralle for
   for (int i = 0; i < RANGE; i++)
      h_1[i] += h_2[i];
}

void random_array(int* arr) {
   omp_set_num_threads(NUM_THREADS);

   #pragma omp parallel
   {
      // Get the thread ID
      int tid = omp_get_thread_num();

      // Calculate the range for each thread
      int start = tid * (SIZE / NUM_THREADS);
      int end = (tid + 1) * (SIZE / NUM_THREADS);

      // Generate and insert random numbers within the thread's range
      for (int i = start; i < end; i++)
         arr[i] = rand() % RANGE;
   }
}

void print_arr(int* arr) {
   for (int i=0; i < RANGE; i++)
      printf("\n %d: %d",i, arr[i]);
}
