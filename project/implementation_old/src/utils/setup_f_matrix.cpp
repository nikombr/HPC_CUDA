#include "../../lib/poisson.h"
#include "../../lib/malloc3d.h"
#include "../../lib/cudaMalloc3d.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <cuda_runtime_api.h>
#ifdef _NORMAL_MAKE
    #include <mpi.h>
#endif
using namespace std;

void Poisson::setup_f_matrix() {
    if (world_rank == 0) { // Allocate and initialse on first process and send to other processes if applicable
        // Allocate on host
        f_h = host_malloc_3d(N + 2, N + 2, N + 2);
        // Check allocation on host
        if (this->f_h == NULL) {
            perror("Allocation of f failed on host!");
            exit(-1);
        }
        // Initialize values
        double delta = 2.0/(this->N+1);
        double fracdelta = (this->N+1)/2.0;
        // Set f to zero everywhere
        memset(f_h[0][0],0,(this->N+2)*(this->N+2)*(this->N+2)*sizeof(double));
        // Overwrite a specific region
        int ux = floor(0.625*fracdelta), uy = floor(0.5*fracdelta), lz = ceil(1.0/3.0*fracdelta), uz = floor(fracdelta);
        for (int i = 1; i <= ux; i++) {
            for (int j = 1; j <= uy; j++) {
                for (int k = lz; k <= uz; k++) {   
                    this->f_h[i][j][k] = 200;
                }
            }
        }
        #ifdef _NORMAL_MAKE
        if (world_rank == 0) { // On mpi call 0, send initialization to the rest
            int tempwidth, widthsum = 1;
            tempwidth = width;
            for (int i = 1; i < world_size; i++) {
                widthsum += tempwidth;
                MPI_Recv(&tempwidth,  1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(**f_h    + (N + 2) * (N + 2) * (widthsum - 1), \
                                    (N + 2) * (N + 2) * (tempwidth + 2), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            }
        }
        #endif
    }
    else { // On remaining mpi calls, receive initialization to the rest
        f_h = host_malloc_3d(width + 2, N + 2, N + 2);
        #ifdef _NORMAL_MAKE
        MPI_Send(&width, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Recv(**f_h,    (N + 2) * (N + 2) * (width + 2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        #endif
    }

    if (GPU) {
        cudaMemcpy(f_log, **f_h, (N + 2) * (N + 2) * (width + 2) * sizeof(double), cudaMemcpyHostToDevice);
        host_free_3d(f_h);
    }

}