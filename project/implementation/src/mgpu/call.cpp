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

    // GPU warm-up (missing)

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
    if (poisson.world_rank == 0)
        printf("%d %d %.5f %.5f %.5e # N iterations time transfer_time error\n", poisson.N, poisson.n, stop, stop_transfer, poisson.tolerance);

    // Finalize 
    poisson.finalize(output_type, output_ext, extra_str);
}