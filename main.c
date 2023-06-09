#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "myProto.h"

/*
Simple MPI+OpenMP+CUDA Integration example
Initially the array of size 4*PART is known for the process 0.
It sends the half of the array to the process 1.
Both processes start to increment members of thier members by 1 - partially with OpenMP, partially with CUDA
The results is send from the process 1 to the process 0, which perform the test to verify that the integration worked properly
*/

int main(int argc, char *argv[])
{
   int size, rank, i;
   int *data;
   MPI_Status status;

   // Seed the random number generator
   srand((unsigned int)time(NULL));

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   if (size != 2)
   {
      printf("Run the example with two processes only\n");
      MPI_Abort(MPI_COMM_WORLD, __LINE__);
   }
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   data = (int *)malloc(SIZE * sizeof(int));
   if (!data) {
      fprintf(stderr, "Error in line %d (failed to initiate array)!\n", __LINE__);
      MPI_Abort(MPI_COMM_WORLD, __LINE__);
   }

   // Seperate the task between 2 processes
   // Master process (process 0)
   if (rank == 0)
   {
      random_array(data); // create random array for histogram
      MPI_Send((data + SIZE / 2), SIZE / 2, MPI_INT, 1, 0, MPI_COMM_WORLD);
   }
   else // slave process (process 1)
   {
      // Allocate memory and receive a half of array from the other process
      data = (int *)malloc((SIZE / 2) * sizeof(int));
      MPI_Recv(data, SIZE / 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
   }

   // On each process - perform a second half of its task with CUDA
   int* hist = (int *)calloc((RANGE), sizeof(int));
   
   if (computeOnGPU(data, SIZE / 2, hist) != 0)
      MPI_Abort(MPI_COMM_WORLD, __LINE__);
   
   // Collect the result on one of processes
   if (rank == 0) {
      int* slave_hist = (int *)malloc(RANGE * sizeof(int));
      MPI_Recv(slave_hist, RANGE, MPI_INT, 1, 0, MPI_COMM_WORLD, &status);
      combine_hist(hist, slave_hist); // combine 2 processes hist
      print_arr(hist); // print final results
   }
   else
      MPI_Send(hist, RANGE, MPI_INT, 0, 0, MPI_COMM_WORLD);

   // Perform a test just to verify that integration MPI + OpenMP + CUDA worked as expected
   if (rank == 0) {
      test(data, SIZE, hist);
   }

   MPI_Finalize();

   return 0;
}