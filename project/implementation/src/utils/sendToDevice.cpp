#include <math.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>
#include "../../lib/poisson.h"
#include <cuda_runtime_api.h>


void Poisson::sendToDevice() {

    //cudaMemcpy(this->uold_log, **this->uold_h, (this->N+2) * (this->N+2) * (this->width+2) * sizeof(double), cudaMemcpyHostToDevice);
    cudaSetDevice(0);
    cudaMemcpy(deviceData[0].uold_log, **uold_h,  (N + 2) * (N + 2) * (deviceData[0].width + 2 - (num_device_per_process > 1)) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceData[0].u_log, **uold_h,  (N + 2) * (N + 2) * (deviceData[0].width + 2 - (num_device_per_process > 1)) * sizeof(double), cudaMemcpyHostToDevice);
    if (num_device_per_process == 2) {
        cudaSetDevice(1);
        cudaMemcpy(deviceData[1].uold_log, **uold_h + (N + 2) * (N + 2) * (deviceData[0].width + 1), \
                                                (N + 2) * (N + 2) * (deviceData[1].width + 1) * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceData[1].u_log, **uold_h + (N + 2) * (N + 2) * (deviceData[0].width + 1), \
                                                (N + 2) * (N + 2) * (deviceData[1].width + 1) * sizeof(double), cudaMemcpyHostToDevice);
    }


}