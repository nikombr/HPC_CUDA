
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>
#ifdef _NORMAL_MAKE
    #include <mpi.h>
#endif
#include "../../lib/poisson.h"
#include <cuda_runtime_api.h>
#include "../../lib/cudaMalloc3d.h"
using namespace std;

                                                                                                                                                                                                                                                                                                                                   

void Poisson::init() {

    if (GPU) {
        if (num_device > 1) output_prefix = "poisson_mgpu";
        else output_prefix = "poisson_gpu";
    }
    else {
        output_prefix = "poisson_cpu";
    }

    if (world_rank == 0) {

        // Initialize values
        double delta = 2.0/(N + 1);
        double fracdelta = (N + 1)/2.0;

        // Initialize uold to start_T
        for (int i = 1; i < N + 1; i++) {
            for (int j = 1; j < N + 1; j++) {
                for (int k = 1; k < N + 1; k++) {
                    uold_h[i][j][k] = start_T;
                    if (!GPU)
                        u_h[i][j][k] = start_T;
                }
            }
        }

        // Set the boundary of uold and u
        for (int i = 0; i < N + 2; i++) {
            for (int j = 0; j < N + 2; j++) {

                uold_h[0][j][i] = 20.0;
                uold_h[N + 1][j][i] = 20.0;

                uold_h[i][0][j] = 0;
                uold_h[i][N + 1][j] = 20.0;

                uold_h[i][j][0] = 20.0;
                uold_h[i][j][N + 1] = 20.0;

                if (!GPU) {
                    u_h[0][j][i] = 20.0;
                    u_h[N + 1][j][i] = 20.0;

                    u_h[i][0][j] = 0;
                    u_h[i][N + 1][j] = 20.0;

                    u_h[i][j][0] = 20.0;
                    u_h[i][j][N + 1] = 20.0;
                }
            }
        }
    }

    #ifdef _NORMAL_MAKE
    if (world_size > 1) { // Multiple nodes, send to places
        if (world_rank == 0) { // On mpi call 0, send initialization to the rest
            int tempwidth, widthsum = 1;
            tempwidth = total_width;
            for (int i = 1; i < world_size; i++) {
                widthsum += tempwidth;
                MPI_Recv(&tempwidth,  1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(**uold_h + (N + 2) * (N + 2) * (widthsum - 1), \
                                    (N + 2) * (N + 2) * (tempwidth + 2), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            }
        }
        else { // On remaining mpi calls, receive initialization to the rest
            MPI_Send(&total_width, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Recv(**uold_h, (N + 2) * (N + 2) * (total_width + 2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, \
                                                                                  MPI_STATUS_IGNORE);
        }
    }
    #endif
}