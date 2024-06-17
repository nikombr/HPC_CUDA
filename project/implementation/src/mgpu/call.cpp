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
    poisson.setupMultipleGPU(false);
    // Setup f matrix
    poisson.setup_f_matrix();
    
    // Allocation
    poisson.alloc();

    // Initialize matrices on host
    poisson.init();

    MPI_Barrier(MPI_COMM_WORLD);

    // GPU warm-up
    poisson.sendToDevice();
    poisson.n = iter_max-30;
    poisson.jacobi();
    poisson.tolerance = tolerance;
    poisson.n = 0;

    MPI_Barrier(MPI_COMM_WORLD);

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
    double temp, tempsum;
    if (poisson.world_rank == 0) {
        printf("%d %d ", poisson.N, poisson.n);
        //printf("%d %d %.5e %.5e %.5e # N iterations time transfer_time error\n", poisson.N, poisson.n, stop, stop_transfer, poisson.tolerance);
        //printf("%.5e ", stop);
        tempsum = stop;
        for (int i = 1; i < poisson.world_size; i++) {
            MPI_Recv(&temp,  1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //printf("%.5e ", temp);
            tempsum += temp;
        }
        printf("%.5e ", tempsum/poisson.world_size);
        //printf("k%.5e ", stop_transfer);
        tempsum = stop_transfer;
        for (int i = 1; i < poisson.world_size; i++) {
            MPI_Recv(&temp,  1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //printf("kk%.5e ", temp);
            tempsum += temp;
        }
        printf("%.5e ", tempsum/poisson.world_size);
        //printf("k%.5e ", poisson.time_nccl_setup);
        tempsum = poisson.time_nccl_setup;
        for (int i = 1; i < poisson.world_size; i++) {
            MPI_Recv(&temp,  1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //printf("kk%.5e ", temp);
            tempsum += temp;
        }
        printf("%.5e ", tempsum/poisson.world_size);
        //printf("%.5e ", poisson.time_nccl_transfer);
        tempsum = poisson.time_nccl_transfer;
        for (int i = 1; i < poisson.world_size; i++) {
            MPI_Recv(&temp,  1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //printf("%.5e ", temp);
            tempsum += temp;
        }
        printf("%.5e ", tempsum/poisson.world_size);
        printf("%.5e ", poisson.time_nccl_transfer);
        tempsum = poisson.time_nccl_transfer;
        for (int i = 1; i < poisson.world_size; i++) {
            MPI_Recv(&temp,  1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("%.5e ", temp);
        }
        printf("%.5e # N iterations time transfer_time nccl_setup nccl_transfer error\n", poisson.tolerance);
    }
    else {
        MPI_Send(&stop, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&stop_transfer, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&poisson.time_nccl_setup, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&poisson.time_nccl_transfer, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&poisson.time_nccl_transfer, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

    }
    
    // Finalize 
    poisson.finalize(output_type, output_ext, extra_str);
}