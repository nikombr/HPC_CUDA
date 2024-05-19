#include <math.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>
#include "../../lib/poisson.h"
#include <cuda_runtime_api.h>


void Poisson::sendToDevice() {

    cudaMemcpy(this->uold_log, **this->uold_h, (this->N+2) * (this->N+2) * (this->width+2) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(this->u_log,    **this->u_h,    (this->N+2) * (this->N+2) * (this->width+2) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(this->f_log,    **this->f_h,    (this->N+2) * (this->N+2) * (this->width+2) * sizeof(double), cudaMemcpyHostToDevice);

}