#include <math.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>
#include "../../lib/poisson.h"
#include <cuda_runtime_api.h>
//#include <mpi.h>


void Poisson::sendToHost() {
    cudaSetDevice(0);
    cudaMemcpy(**uold_h, deviceData[0].uold_log,  (N + 2) * (N + 2) * (deviceData[0].width + 2 - (num_device_per_process > 1)) * sizeof(double), cudaMemcpyDeviceToHost);
    if (num_device_per_process == 2) {
        cudaDeviceSynchronize();
        cudaSetDevice(1);
        cudaMemcpy(**uold_h + (N + 2) * (N + 2) * (deviceData[0].width + 1), deviceData[1].uold_log, \
                                                (N + 2) * (N + 2) * (deviceData[1].width + 1) * sizeof(double), cudaMemcpyDeviceToHost);
    }


    
}