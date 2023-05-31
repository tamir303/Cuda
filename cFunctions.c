#include "myProto.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
    printf("The test passed successfully\n"); 
}
