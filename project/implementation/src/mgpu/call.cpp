#include <stdio.h>
#include <stdlib.h>
#include "../../lib/poisson.h"
#include <omp.h>
#include <mpi.h>
#include <cuda_runtime_api.h>


void call(int N, int output_type, char *output_prefix, char*output_ext, char*extra_str, \
                        double tolerance, int iter_max, double start_T) {

    // Initialize
    Poisson poisson = Poisson(N, true, start_T, iter_max, tolerance);

    // Setup multiple GPUs
    poisson.setupMultipleGPU(true);

    // Allocation
    poisson.alloc();

    // Initialize matrices on host
    poisson.init();

    // GPU warm-up
    poisson.sendToDevice();
    poisson.jacobi();
    poisson.tolerance = tolerance;
    poisson.n = 0;

    double start_transfer = omp_get_wtime();
    // Copy matrices to device
    poisson.sendToDevice();

    double start = omp_get_wtime();
    // Run Jacobi iterations
    poisson.jacobi();
    double stop = omp_get_wtime() - start;

    // Copy matrix uold back to host
    poisson.sendToHost();
    double stop_transfer = omp_get_wtime() - start_transfer;
    MPI_Barrier(MPI_COMM_WORLD);
    double tempstop, tempstop_transfer;
    if (poisson.world_rank == 0) {
        printf("%d %d ", poisson.N, poisson.n);
        //printf("%d %d %.5e %.5e %.5e # N iterations time transfer_time error\n", poisson.N, poisson.n, stop, stop_transfer, poisson.tolerance);
        printf("%.5e ", stop);
        for (int i = 1; i < poisson.world_rank; i++) {
            MPI_Recv(&tempstop,  1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("%.5e ", tempstop);
        }
        printf("%.5e ", stop_transfer);
        for (int i = 1; i < poisson.world_rank; i++) {
            MPI_Recv(&tempstop_transfer,  1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("%.5e ", tempstop_transfer);
        }
    }
    else {
        MPI_Send(&stop, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&stop_transfer, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

    }

    // Finalize 
    poisson.finalize(output_type, output_ext, extra_str);
}