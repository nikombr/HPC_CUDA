#include <math.h>
#include <stdio.h>
#include <omp.h>
#include "../../../lib/iteration_mgpu.h"
#include <mpi.h>
#include "nccl.h"
#include "../../../lib/poisson.h"

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

void Poisson::jacobi() {

    double start, stop = 0;
    start = omp_get_wtime();

    // Setup nccl
    ncclGroupStart();
    ncclUniqueId id;
    if (this->world_rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, this->world_size, id, this->world_rank));
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    ncclGroupEnd();

    this->time_nccl_setup = omp_get_wtime() - start;

    // Get constants for nccl sendings
    int firstRowReceive = 0;
    int lastRowSend     = (N + 2) * (N + 2) * (width);
    int firstRowSend    = firstRowReceive + (N + 2) * (N + 2);
    int lastRowReceive  = lastRowSend + (N + 2) * (N + 2);
    
    while ((this->n < this->iter_max)) {
        // Do iteration
        iteration(this->u_d, this->uold_d, this->f_d, this->N, this->iter_max, this->width);
        
        
        // Send data from different MPI calls with nccl
        start = omp_get_wtime();
        ncclGroupStart(); // Start nccl up
        if (this->world_rank == 0) { // We only need to send to and receive from the next device
            // Rank 0 receives data from rank 1
            NCCLCHECK(ncclRecv(this->u_log + lastRowReceive, (N + 2) * (N + 2), ncclDouble, this->world_rank + 1, comm, stream));
            // Rank 0 sends data to rank 1
            NCCLCHECK(ncclSend(this->u_log + lastRowSend,    (N + 2) * (N + 2), ncclDouble, this->world_rank + 1, comm, stream));
        }
        else if (this->world_rank + 1 == this->world_size) { // We only need to send to and receive from the previous device
            // Rank m sends data to rank m - 1
            NCCLCHECK(ncclSend(this->u_log + firstRowSend,    (N + 2) * (N + 2), ncclDouble, this->world_rank - 1, comm, stream));
            // Rank m receives data from rank m - 1
            NCCLCHECK(ncclRecv(this->u_log + firstRowReceive, (N + 2) * (N + 2), ncclDouble, this->world_rank - 1, comm, stream));
        }
        else { // We need to send to and receive from both the previous and next device
            // Rank i sends data to rank i - 1
            NCCLCHECK(ncclSend(this->u_log + firstRowSend,    (N + 2) * (N + 2), ncclDouble, this->world_rank - 1, comm, stream));
            // Rank i receives data from rank i - 1
            NCCLCHECK(ncclRecv(this->u_log + firstRowReceive, (N + 2) * (N + 2), ncclDouble, this->world_rank - 1, comm, stream));
            // Rank i receives data from rank i + 1
            NCCLCHECK(ncclRecv(this->u_log + lastRowReceive,  (N + 2) * (N + 2), ncclDouble, this->world_rank + 1, comm, stream));
            // Rank i sends data to rank i + 1
            NCCLCHECK(ncclSend(this->u_log + lastRowSend,     (N + 2) * (N + 2), ncclDouble, this->world_rank + 1, comm, stream));

        }
        ncclGroupEnd(); // End nccl
        stop += omp_get_wtime() - start;
        
        
        // Swap addresses
        this->swapArrays();
        // Next iteration
        (this->n)++;
    }
    
    this->time_nccl_transfer = stop;
    return;
}
