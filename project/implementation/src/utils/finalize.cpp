#include <math.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>
#include <cuda_runtime_api.h>
#include <mpi.h>

#include "../../lib/poisson.h"
#include "../../lib/malloc3d.h"
#include "../../lib/cudaMalloc3d.h"
#include "../../lib/dumpOutput.h"


void Poisson::finalize(int output_type, char*output_ext, char *extra_str) {

    if (this->GPU && this->width < this->N) {
        
        MPI_Barrier(MPI_COMM_WORLD);
        // Send stuff with MPI
        if (this->world_rank == 0) {
            int tempwidth, widthsum;
            tempwidth = this->width;
            widthsum = 1;
            //printf("mpi 0: tempwidth = %d, widthsum = %d \n",tempwidth,widthsum);
            for (int i = 1; i < this->world_size; i++) {
                widthsum += tempwidth;
                MPI_Recv(&tempwidth,  1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                //printf("mpi %d: tempwidth = %d, widthsum = %d \n",i,tempwidth,widthsum);
                MPI_Recv(**this->uold_h + (this->N+2) * (this->N+2) * widthsum, (this->N+2) * (this->N+2) * (tempwidth), MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                MPI_Recv(**this->uold_h + (this->N+2) * (this->N+2) * widthsum, (this->N+2) * (this->N+2) * (tempwidth), MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            
        }
        else {
            //printf("i am sending width = %d from mpi 1\n",this->width);
            MPI_Send(&this->width, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(**this->uold_h + (this->N + 2) * (this->N + 2), (this->N + 2) * (this->N + 2) * (this->width), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            MPI_Send(**this->uold_h + (this->N + 2) * (this->N + 2), (this->N + 2) * (this->N + 2) * (this->width), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        FILE *fp;
    if (this->world_rank == 0) {
    fp = fopen("results/dev0.txt", "w");
             if (fp!= NULL) {
                double delta = 2.0/(N+1);
            for (int i = 0; i < N+2; i++) {
                for (int j = 0; j < N+2; j++) {
                    for (int k = 0; k < N+2; k++) {
                        fprintf(fp,"\t%.2f ",this->uold_h[i][j][k]);

                    }
                    fprintf(fp,"\n");

                }
                fprintf(fp,"\n\n");
            }
            }
    } else {
        if (this->world_rank == 1) fp = fopen("results/dev1.txt", "w");
        if (this->world_rank == 2) fp = fopen("results/dev2.txt", "w");
        if (this->world_rank == 3) fp = fopen("results/dev3.txt", "w");
             if (fp!= NULL) {
                double delta = 2.0/(N+1);
            for (int i = 0; i < width+2; i++) {
                for (int j = 0; j < N+2; j++) {
                    for (int k = 0; k < N+2; k++) {
                        fprintf(fp,"\t%.2f ",this->uold_h[i][j][k]);

                    }
                    fprintf(fp,"\n");

                }
                fprintf(fp,"\n\n");
            }
            }
    }

    }
    

    // Dump results if wanted (if multiple MPI then only on first)
    if (this->world_rank == 0) {
        dump_output(output_type, this->output_prefix, output_ext, extra_str, this->N, this->uold_h);
    }

    // De-allocate memory
    if (this->GPU) {
        if (this->width != this->N) MPI_Finalize();
        host_free_3d(this->u_h);
        host_free_3d(this->f_h);
        host_free_3d(this->uold_h);
        device_free_3d(this->u_d,    this->u_log);
        device_free_3d(this->f_d,    this->f_log);
        device_free_3d(this->uold_d, this->uold_log);
    }
    else {
        free_3d(this->u_h);
        free_3d(this->f_h);
        free_3d(this->uold_h);
    }

    
}