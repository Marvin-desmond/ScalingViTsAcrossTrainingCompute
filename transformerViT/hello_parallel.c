#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, world_size;
    int values[4] = {39, 72, 129, 42};
    int scattered_val;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Scatter(&values, 1, MPI_INT, &scattered_val, 1, MPI_INT, 0, MPI_COMM_WORLD);
    char *s_or_p_processor = world_size < 2 ? "processor" : "processors";
    printf("Processor %d received %d\n",rank, scattered_val);
    MPI_Barrier(MPI_COMM_WORLD);
    // a computation to simulate a forward and backward pass 
    scattered_val = scattered_val * 0.5; 
    printf("Processor %d updated value %d\n",rank, scattered_val);
    MPI_Barrier(MPI_COMM_WORLD);
    // gather to rank zero
    int global_avg;
    MPI_Reduce(&scattered_val, &global_avg, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0){
        printf("Rank %d: sum %d average %.1f\n", 
            rank, global_avg, (float)global_avg/world_size);
    }
    // Broadcast the result to all the processes
    float broadcast_val = 1.0f;
    if (rank == 0){
        broadcast_val = (float)global_avg/world_size;
    }
    MPI_Bcast(&broadcast_val, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    printf("Processor %d final value %f\n",rank, broadcast_val);
    MPI_Finalize();
}