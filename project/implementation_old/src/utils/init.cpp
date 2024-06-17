
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

    if (this->GPU) {
        if (this->width < this->N) this->output_prefix = "poisson_mgpu";
        else this->output_prefix = "poisson_gpu";
    }
    else {
        this->output_prefix = "poisson_cpu";
    }

    if (this->world_rank == 0) {

        // Initialize values
        double delta = 2.0/(this->N+1);
        double fracdelta = (this->N+1)/2.0;

        // Initialize uold to start_T
        for (int i = 1; i < this->N+1; i++) {
            for (int j = 1; j < this->N+1; j++) {
                for (int k = 1; k < this->N+1; k++) {
                    this->uold_h[i][j][k] = start_T;
                    if (!(this->GPU))
                        this->u_h[i][j][k] = start_T;
                }
            }
        }

        // Set the boundary of uold and u
        for (int i = 0; i < this->N+2; i++) {
            for (int j = 0; j < this->N+2; j++) {

                this->uold_h[0][j][i] = 20.0;
                this->uold_h[this->N+1][j][i] = 20.0;

                this->uold_h[i][0][j] = 0;
                this->uold_h[i][this->N+1][j] = 20.0;

                this->uold_h[i][j][0] = 20.0;
                this->uold_h[i][j][this->N+1] = 20.0;

                if (!(this->GPU)) {
                    this->u_h[0][j][i] = 20.0;
                    this->u_h[this->N+1][j][i] = 20.0;

                    this->u_h[i][0][j] = 0;
                    this->u_h[i][this->N+1][j] = 20.0;

                    this->u_h[i][j][0] = 20.0;
                    this->u_h[i][j][this->N+1] = 20.0;
                }
            }
        }
    }

    #ifdef _NORMAL_MAKE
    if (this->width < this->N) { // Multiple GPUs, send to places
        if (world_rank == 0) { // On mpi call 0, send initialization to the rest
            int tempwidth, widthsum = 1;
            tempwidth = this->width;
            for (int i = 1; i < world_size; i++) {
                widthsum += tempwidth;
                MPI_Recv(&tempwidth,  1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(**this->uold_h + (N + 2) * (N + 2) * (widthsum - 1), \
                                    (N + 2) * (N + 2) * (tempwidth + 2), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                //MPI_Send(**this->u_h    + (N + 2) * (N + 2) * (widthsum - 1), \
                                    (N + 2) * (N + 2) * (tempwidth + 2), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                //MPI_Send(**this->f_h    + (N + 2) * (N + 2) * (widthsum - 1), \
                                    (N + 2) * (N + 2) * (tempwidth + 2), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            }
        }
        else { // On remaining mpi calls, receive initialization to the rest
            MPI_Send(&this->width, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Recv(**this->uold_h, (N + 2) * (N + 2) * (width + 2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, \
                                                                                  MPI_STATUS_IGNORE);
            //MPI_Recv(**this->u_h,    (N + 2) * (N + 2) * (width + 2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, \
                                                                                  MPI_STATUS_IGNORE);
            //MPI_Recv(**this->f_h,    (N + 2) * (N + 2) * (width + 2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, \
                                                                                  MPI_STATUS_IGNORE);
        }
    }
    #endif
}