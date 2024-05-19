#include <math.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>
#include "../../lib/poisson.h"
#include <cuda_runtime_api.h>
#include <mpi.h>


void Poisson::sendToHost() {

    cudaMemcpy(**this->uold_h, this->uold_log, (this->N+2) * (this->N+2) * (this->width+2) * sizeof(double), cudaMemcpyDeviceToHost);

    
}