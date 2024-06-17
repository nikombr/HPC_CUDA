#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "../../lib/poisson.h"
#include <cuda_runtime_api.h>
#include <omp.h>

void Poisson::setupMultipleGPU(int print) {
    MPI_Init(NULL, NULL);
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // Get number of devices accessible from the node where this rank is located
	cudaGetDeviceCount(&num_device_per_process);
    // Get number of total GPUs in system
    num_device = num_device_per_process * world_size;
    // Get section we need on this specific device (integer division)
    for (int i = 0; i < num_device_per_process; i++) {
        deviceData[i].rank  = i + num_device_per_process * world_rank;
        deviceData[i].first = deviceData[i].rank == 0;
        deviceData[i].last  = deviceData[i].rank == num_device - 1;
        deviceData[i].start = N/num_device * deviceData[i].rank + 1;
        deviceData[i].end   = deviceData[i].last ?  N : N/num_device*(deviceData[i].rank + 1);
        deviceData[i].width = deviceData[i].end  - deviceData[i].start + 1;
        total_width += deviceData[i].width;
        cudaSetDevice(i);
        cudaDeviceEnablePeerAccess (1-i, 0);
    }
    if (num_device_per_process > 1) {
        deviceData[0].peer_width = deviceData[1].width;
        deviceData[1].peer_width = deviceData[0].width;
        deviceData[0].canAccesPeerNext = true;
        deviceData[1].canAccesPeerPrev = true;
    }
    // Get constants for nccl transfers
    for (int i = 0; i < num_device_per_process; i++) {
        deviceData[i].firstRowReceive = 0;
        deviceData[i].lastRowSend     = (N + 2) * (N + 2) * (deviceData[i].width  - deviceData[i].canAccesPeerPrev);
        deviceData[i].firstRowSend    = deviceData[i].firstRowReceive + (N + 2) * (N + 2);
        deviceData[i].lastRowReceive  = deviceData[i].lastRowSend + (N + 2) * (N + 2);
    }

    // Print information to checkk everything is as it should
    if (print) {
        int temp_start, temp_end, temp_num_device_per_process;
        // Show information about MPI call on device 0 by sending everything to rank 0
        if (this->world_rank == 0) {
            // Print for rank 0
            printf("Rank 0: I can see %d devices!\n",num_device_per_process);
            for (int i = 0; i < num_device_per_process; i++) {
                printf("\t(start, end) = (%d, %d)\n",deviceData[i].start,deviceData[i].end);
            }
            // Receive from other ranks and print
            for (int j = 1; j < world_size; j++) {
                MPI_Recv(&temp_num_device_per_process, 1, MPI_INT, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                printf("Rank %d: I can see %d devices!\n",j,temp_num_device_per_process);
                for (int i = 0; i < temp_num_device_per_process;  i++) {
                    MPI_Recv(&temp_start,             1, MPI_INT, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&temp_end,               1, MPI_INT, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    printf("\t(start, end) = (%d, %d)\n",temp_start,temp_end);
                }
            }
        }
        else {
            // Send to rank 0
            MPI_Send(&num_device_per_process, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            for (int i = 0; i < num_device_per_process; i++) {
                MPI_Send(&deviceData[i].start,             1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                MPI_Send(&deviceData[i].end,               1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

    }
}