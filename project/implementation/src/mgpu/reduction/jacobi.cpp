#include <math.h>
#include <stdio.h>
#include <omp.h>
#include "../../../lib/iteration_mgpu_reduction.h"
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
    ncclComm_t * comms;
    cudaStream_t * streams;
    streams = (cudaStream_t*) malloc(sizeof(cudaStream_t) * num_device_per_process);
    if (world_size > 1) {
        // Setup nccl
        NCCLCHECK(ncclGroupStart());
        ncclUniqueId id;
        if (world_rank == 0) ncclGetUniqueId(&id);
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        
        comms     = (ncclComm_t*)   malloc(sizeof(ncclComm_t)   * num_device_per_process);
        
        for (int i = 0; i < num_device_per_process; i++) {
            cudaSetDevice(i);
            NCCLCHECK(ncclCommInitRank(comms + i, num_device_per_process * world_size, id, world_rank * num_device_per_process + i));
            cudaStreamCreate(streams + i);
        }
        NCCLCHECK(ncclGroupEnd());
    }
    else {
        for (int i = 0; i < num_device_per_process; i++) {
            cudaSetDevice(i);
            cudaStreamCreate(streams + i);
        }
    }

    this->time_nccl_setup = omp_get_wtime() - start;


    // Setup sum for reduction
    double  *sum_h, *sum_d_1, *sum_d_2, *sum_bound_h;
    double *sum[2], *sum_bound[2];
    cudaMallocHost((void**)&sum_h, num_device_per_process*sizeof(double));
    cudaMalloc((void**)&sum_d_1, sizeof(double));
    cudaMalloc((void**)&sum_d_2, sizeof(double));
    cudaMallocHost((void**)&sum_bound_h, num_device_per_process*sizeof(double));
    sum[0] = sum_d_1;
    sum[1] = sum_d_2;
    double tempsum, sum_all;
    sum_all = tolerance + 1;

    while (n < iter_max && sum_all > tolerance) {
        sum_all = 0;
        //#pragma omp parallel for schedule(static)
        for (int i = 0; i < num_device_per_process; i++) {
            cudaSetDevice(i);
            iteration(deviceData[i].u_d, deviceData[i].uold_d, deviceData[1-i].uold_d, deviceData[i].f_d, N, sum[i], deviceData[i].width, deviceData[i].peer_width,deviceData[i].canAccesPeerPrev,deviceData[i].canAccesPeerNext,streams[i]);
            cudaMemcpyAsync(&sum_h[i], sum[i], sizeof(double), cudaMemcpyDeviceToHost, streams[i]);
            cudaDeviceSynchronize();
            sum_all += sum_h[i];
        }
        
        start = omp_get_wtime();
        if (world_size > 1) {
            if (world_rank == 0) {
                for (int i = 1; i < world_size; i++) {
                    MPI_Recv(&tempsum, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    sum_all += tempsum;
                }
            }
            else {
                MPI_Send(&sum_all, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
            MPI_Bcast(&sum_all, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        // Send data from different MPI calls with nccl
       
        if (world_size > 1) {
            NCCLCHECK(ncclGroupStart()); // Start nccl up
            //#pragma omp parallel for schedule(static)
            for (int i = 0; i < num_device_per_process; i++) {
                cudaSetDevice(i);
                if (deviceData[i].rank > 0 && !deviceData[i].canAccesPeerPrev) {
                    //printf("Lower boundary from %d on MPI-call %d\n",deviceData[i].rank,world_rank);
                    // Rank m sends data to rank m - 1
                    NCCLCHECK(ncclSend(deviceData[i].u_log + deviceData[i].firstRowSend,    (N + 2) * (N + 2), ncclDouble, world_rank * num_device_per_process + i - 1, comms[i], streams[i]));
                    // Rank m receives data from rank m - 1
                    NCCLCHECK(ncclRecv(deviceData[i].u_log + deviceData[i].firstRowReceive, (N + 2) * (N + 2), ncclDouble, world_rank * num_device_per_process + i - 1, comms[i], streams[i]));
                }
                if (deviceData[i].rank < num_device_per_process * world_size - 1 && !deviceData[i].canAccesPeerNext) {
                    //printf("Upper boundary from %d on MPI-call %d\n",deviceData[i].rank,world_rank);
                    // Rank i receives data from rank i + 1
                    NCCLCHECK(ncclRecv(deviceData[i].u_log + deviceData[i].lastRowReceive,  (N + 2) * (N + 2), ncclDouble, world_rank * num_device_per_process + i + 1, comms[i], streams[i]));
                    // Rank i sends data to rank i + 1
                    NCCLCHECK(ncclSend(deviceData[i].u_log + deviceData[i].lastRowSend,     (N + 2) * (N + 2), ncclDouble, world_rank * num_device_per_process + i + 1, comms[i], streams[i]));
                }
            }
            NCCLCHECK(ncclGroupEnd()); // End nccl
        }
        stop += omp_get_wtime() - start;
        
        // Swap addresses
        swapArrays();
        // Next iteration
        n++;
    }
    
    this->time_nccl_transfer = stop;
    if (world_rank == 0) {
        tolerance = sum_all;
    }
    return;
}
