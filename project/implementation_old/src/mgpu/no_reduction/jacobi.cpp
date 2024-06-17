#include <math.h>
#include <stdio.h>
#include <omp.h>
#include "../../../lib/iteration_mgpu.h"
#include <mpi.h>
#include "nccl.h"
#include "../../../lib/poisson.h"
#include <cuda_runtime_api.h>

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

    // Get constants for nccl transfers
    int firstRowReceive = 0;
    int lastRowSend     = (N + 2) * (N + 2) * (width);
    int firstRowSend    = firstRowReceive + (N + 2) * (N + 2);
    int lastRowReceive  = lastRowSend + (N + 2) * (N + 2);
    
    while ((this->n < this->iter_max)) {
        // Do iteration
        iteration(this->u_d, this->uold_d, this->f_d, this->N, this->width);
        // Send data from different MPI calls with nccl and measure the time
        start = omp_get_wtime();
        ncclGroupStart(); // Start nccl up
        if (this->world_size > 1) {
            if (world_rank > 0) {
                // Rank m sends data to rank m - 1
                NCCLCHECK(ncclSend(u_log + firstRowSend,    (N + 2) * (N + 2), ncclDouble, world_rank - 1, comm, stream));
                // Rank m receives data from rank m - 1
                NCCLCHECK(ncclRecv(u_log + firstRowReceive, (N + 2) * (N + 2), ncclDouble, world_rank - 1, comm, stream));
            }
            if (world_rank + 1 < world_size) {
                // Rank i receives data from rank i + 1
                NCCLCHECK(ncclRecv(u_log + lastRowReceive,  (N + 2) * (N + 2), ncclDouble, world_rank + 1, comm, stream));
                // Rank i sends data to rank i + 1
                NCCLCHECK(ncclSend(u_log + lastRowSend,     (N + 2) * (N + 2), ncclDouble, world_rank + 1, comm, stream));
            }
        }
        ncclGroupEnd(); // End nccl
        //cudaDeviceSynchronize();
        stop += omp_get_wtime() - start;
        
        
        
        // Swap addresses
        this->swapArrays();
        // Next iteration
        (this->n)++;
    }
    // Save nccl transfer time
    this->time_nccl_transfer = stop;
    return;
}
