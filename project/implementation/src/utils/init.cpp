
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>
#include <mpi.h>
#include "../../lib/poisson.h"
using namespace std;

                                                                                                                                                                                                                                                                                                                                   

void Poisson::init() {

    if (this->GPU) {
        if (this->width < this->N) this->output_prefix = "poisson_mgpu";
        else this->output_prefix = "poisson_gpu";
    }
    else {
        this->output_prefix = "poisson_cpu";
    }

    //if (this->width == this->N) {
    if (this->world_rank == 0) {

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

        // Initialize uold to start_T
        for (int i = 1; i < this->N+1; i++) {
            for (int j = 1; j < this->N+1; j++) {
                for (int k = 1; k < this->N+1; k++) {
                    this->uold_h[i][j][k] = start_T;
                    this->u_h[i][j][k] = start_T;
                }
            }
        }

        // Set the boundary of uold and u
        for (int i = 0; i < this->N+2; i++) {
            for (int j = 0; j < this->N+2; j++) {

                this->uold_h[0][j][i] = 20.0;
                this->uold_h[this->N+1][j][i] = 20.0;

                this->u_h[0][j][i] = 20.0;
                this->u_h[this->N+1][j][i] = 20.0;

                this->uold_h[i][0][j] = 0;
                this->uold_h[i][this->N+1][j] = 20.0;

                this->u_h[i][0][j] = 0;
                this->u_h[i][this->N+1][j] = 20.0;

                this->uold_h[i][j][0] = 20.0;
                this->uold_h[i][j][this->N+1] = 20.0;

                this->u_h[i][j][0] = 20.0;
                this->u_h[i][j][this->N+1] = 20.0;
            }
        }
    }


    if (this->width < this->N) { // Multiple GPUs, send to places

        // Get width on different devices
        //int *widths = malloc(this->world_size*sizeof(int));
        //widths[0] = this->width;
        
        MPI_Barrier(MPI_COMM_WORLD);
        // On mpi call 0, send initialization to the rest
        if (this->world_rank == 0) {
            int tempwidth, widthsum;
            tempwidth = this->width;
            widthsum = 1;
            //printf("mpi 0: tempwidth = %d, widthsum = %d \n",tempwidth,widthsum);
            for (int i = 1; i < this->world_size; i++) {
                widthsum += tempwidth;
                
                MPI_Recv(&tempwidth,  1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                //printf("mpi %d: tempwidth = %d, widthsum = %d \n",i,tempwidth,widthsum);
                MPI_Send(**this->uold_h + (this->N+2) * (this->N+2) * (widthsum - 1), (this->N+2) * (this->N+2) * (tempwidth + 2), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                MPI_Send(**this->u_h + (this->N+2) * (this->N+2) * (widthsum - 1), (this->N+2) * (this->N+2) * (tempwidth + 2), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                MPI_Send(**this->f_h + (this->N+2) * (this->N+2) * (widthsum - 1), (this->N+2) * (this->N+2) * (tempwidth + 2), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);

            }

        }
        else {
            
            MPI_Send(&this->width, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Recv(**this->uold_h, (this->N+2) * (this->N+2) * (this->width + 2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(**this->u_h, (this->N+2) * (this->N+2) * (this->width + 2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(**this->f_h, (this->N+2) * (this->N+2) * (this->width + 2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}