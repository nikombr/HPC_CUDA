#include <math.h>
#include <stdio.h>
#include <omp.h>
#include "../../../lib/iteration_mgpu_reduction.h"
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

    double start_loop, start, stop = 0;
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

    start_loop = omp_get_wtime();

    // Setup sum for reduction
    double  *sum_h, *sum_d;
    cudaMallocHost((void**)&sum_h, sizeof(double));
    cudaMalloc((void**)&sum_d, sizeof(double));
    double tempsum, sum_all;
    sum_all = this->tolerance + 1;

    while ((this->n < this->iter_max && sum_all > this->tolerance)) {
        *sum_h = 0.0;
        cudaMemcpy(sum_d, sum_h, sizeof(double), cudaMemcpyHostToDevice);

        // Do iteration
        iteration(this->u_d, this->uold_d, this->f_d, this->N, this->width, sum_d);

        cudaMemcpy(sum_h, sum_d, sizeof(double), cudaMemcpyDeviceToHost);
        if (this->world_rank == 0) {
            sum_all = *sum_h;
            for (int i = 1; i < this->world_size; i++) {
                MPI_Recv(&tempsum, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                sum_all += tempsum;
            }
            for (int i = 1; i < this->world_size; i++) {
                MPI_Send(&sum_all, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            }
        }
        else {
            MPI_Send(sum_h, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            MPI_Recv(&sum_all, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // Send data from different MPI calls with nccl
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
    
    this->time_nccl_transfer = stop;
    if (world_rank == 0) {
        this->tolerance = sum_all;
    }
    return;
}
