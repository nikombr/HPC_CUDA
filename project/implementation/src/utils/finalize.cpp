#include <math.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>
//#include <cuda_runtime_api.h>
#ifdef _NORMAL_MAKE
    #include <mpi.h>
#endif

#include "../../lib/poisson.h"
#include "../../lib/malloc3d.h"
#include "../../lib/cudaMalloc3d.h"
#include "../../lib/dumpOutput.h"
#include <cuda_runtime_api.h>

void Poisson::finalize(int output_type, char*output_ext, char *extra_str) {

    #ifdef _NORMAL_MAKE
    if (GPU && world_size > 1) {
        
        MPI_Barrier(MPI_COMM_WORLD);
        // Send stuff with MPI
        if (world_rank == 0) {
            int tempwidth, widthsum;
            tempwidth = total_width;
            widthsum = 1;
            //printf("mpi 0: tempwidth = %d, widthsum = %d \n",tempwidth,widthsum);
            for (int i = 1; i < world_size; i++) {
                widthsum += tempwidth;
                MPI_Recv(&tempwidth,  1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                //printf("mpi %d: tempwidth = %d, widthsum = %d \n",i,tempwidth,widthsum);
                MPI_Recv(**uold_h + (N + 2) * (N + 2) * widthsum, (N + 2) * (N + 2) * (tempwidth + 1), MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                MPI_Recv(**uold_h + (N + 2) * (N + 2) * widthsum, (N + 2) * (N + 2) * (tempwidth + 1), MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        else {
            //printf("i am sending width = %d from mpi 1\n",this->width);
            MPI_Send(&total_width, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(**uold_h + (N + 2) * (N + 2), (N + 2) * (N + 2) * (total_width + 1), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            MPI_Send(**uold_h + (N + 2) * (N + 2), (N + 2) * (N + 2) * (total_width + 1), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);


    }
    #endif
    

    // Dump results if wanted (if multiple MPI then only on first)
    if (this->world_rank == 0) {
        dump_output(output_type, output_prefix, output_ext, extra_str, N, uold_h);
    }

    // De-allocate memory
    if (GPU) {
        #ifdef _NORMAL_MAKE
        if (num_device > 1) MPI_Finalize();
        #endif
        //host_free_3d(this->u_h);
        //host_free_3d(this->f_h);
        host_free_3d(uold_h);
        for (int i = 0; i < num_device_per_process; i++) {
            cudaSetDevice(i);
            device_free_3d(deviceData[i].u_d,    deviceData[i].u_log);
            device_free_3d(deviceData[i].f_d,    deviceData[i].f_log);
            device_free_3d(deviceData[i].uold_d, deviceData[i].uold_log);
        }
    }
    else {
        host_free_3d(u_h);
        host_free_3d(f_h);
        host_free_3d(uold_h);
    }

    
}