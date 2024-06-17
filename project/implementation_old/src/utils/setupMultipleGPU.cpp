#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "../../lib/poisson.h"
#include <cuda_runtime_api.h>
#include <omp.h>

void Poisson::setupMultipleGPU(int print) {
    MPI_Init(NULL, NULL);
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &this->world_size);
    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &this->world_rank);
    // Get number of devices accessible from the node where this rank is located
	cudaGetDeviceCount(&this->num_device);
    // Get local rank
    this->local_rank = this->world_rank % this->num_device;
    cudaSetDevice(this->local_rank);
    // Get section we need on this specific device (integer division)
    this->start = this->N/this->world_size*this->world_rank + 1;
    this->end   = this->world_rank == this->world_size - 1 ? this->N : this->N/this->world_size*(this->world_rank + 1);
    this->width = this->end - this->start + 1;

    char host_name[MPI_MAX_PROCESSOR_NAME];
    int len;
    MPI_Get_processor_name( host_name, &len );
    int host_num = (host_name[8]-'0')*10+(host_name[9]-'0');
    
    // Check if we have peer access to next and previous device.
    /*if (this->world_rank < this->world_size - 1) {
        cudaDeviceCanAccessPeer(&this->canAccessPeerNext, this->world_rank, this->world_rank + 1);
    }
    if (world_rank > 0) {
        cudaDeviceCanAccessPeer(&this->canAccessPeerPrev, this->world_rank, this->world_rank - 1);
    }*/
    int host_num_temp;
    if (this->world_size > 1) {
        if (this->world_rank == 0) { // We only need to send to and receive from the next device
            // Rank 0 receives data from rank 1
            MPI_Recv(&host_num_temp, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);            
            // Rank 0 sends data to rank 1
            MPI_Send(&host_num,      1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            if (host_num_temp == host_num) {
                canAccessPeerNext = true;
            }
        }
        else if (this->world_rank + 1 == this->world_size) { // We only need to send to and receive from the previous device
            // Rank m sends data to rank m - 1
            MPI_Send(&host_num,      1, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD);
            // Rank m receives data from rank m - 1
            MPI_Recv(&host_num_temp, 1, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  
            if (host_num_temp == host_num) {
                canAccessPeerPrev = true;
            }
        }
        else { // We need to send to and receive from both the previous and next device
            // Rank i sends data to rank i - 1
            MPI_Send(&host_num,      1, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD);
            // Rank i receives data from rank i - 1
            MPI_Recv(&host_num_temp, 1, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
            if (host_num_temp == host_num) {
                canAccessPeerPrev = true;
            } 
            // Rank i receives data from rank i + 1
            MPI_Recv(&host_num_temp, 1, MPI_INT, world_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  
            // Rank i sends data to rank i + 1
            MPI_Send(&host_num,      1, MPI_INT, world_rank + 1, 0, MPI_COMM_WORLD);
            if (host_num_temp == host_num) {
                canAccessPeerNext = true;
            }
        }
        }
        
    if (print) {
        printf( "Node number %d/%d is %s, %d\n", world_rank, world_size, host_name, host_num );
        int tempstart, tempend, tempCanAccessPeerNext, tempCanAccessPeerPrev;
        // Show information about MPI call on device 0 by sending everything to rank 0
        if (this->world_rank == 0) {
            // Print for rank 0
            printf("Rank 0:\n \t(start, end) = (%d, %d)\n",this->start,this->end);
            if (this->canAccessPeerNext) printf("\tI have peer-access to next device!\n"); // tjek af peer access skal fikses
            printf("\tI can see %d devices!\n",num_device);
            // Receive from other ranks and print
            for (int i = 1; i < world_size; i++) {
                MPI_Recv(&tempstart,             1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&tempend,               1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&tempCanAccessPeerPrev, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&tempCanAccessPeerNext, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                printf("Rank %d:\n \t(start, end) = (%d,%d)\n",i,tempstart,tempend);
                if (tempCanAccessPeerPrev) printf("\tI have peer-access to previous device!\n");
                if (tempCanAccessPeerNext) printf("\tI have peer-access to next device!\n");
            }
        }
        else {
            // Send to rank 0
            MPI_Send(&this->start,             1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&this->end,               1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&this->canAccessPeerPrev, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&this->canAccessPeerNext, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);

    }
}