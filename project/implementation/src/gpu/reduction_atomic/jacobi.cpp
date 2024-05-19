#include <math.h>
#include <stdio.h>
#include <omp.h>
#include "../../../lib/poisson.h"
#include "../../../lib/iteration_reduction.h"
#include <cuda_runtime_api.h>


void Poisson::jacobi() {

    double *sum_h, *sum_d;
    
    cudaMallocHost((void **)&sum_h, sizeof(double));
    cudaMalloc((void **)&sum_d, sizeof(double));
    *sum_h = this->tolerance + 1;
   
    //printf("%d, %d, %d\n",dimGrid.x,dimGrid.y,dimGrid.z);
    while (this->n < iter_max && *sum_h > this->tolerance) {
        *sum_h = 0.0;
        cudaMemcpy(sum_d, sum_h, sizeof(double), cudaMemcpyHostToDevice);
 
        // Do iteration
        iteration(this->u_d, this->uold_d, this->f_d, this->N, this->iter_max, sum_d);

        cudaMemcpy(sum_h, sum_d, sizeof(double), cudaMemcpyDeviceToHost);
        //printf("%.5e, %.5e\n",*sum_h,*tolerance);
        // Swap addresses
        this->swapArrays();
        // Next iteration
        (this->n)++;
    }
    this->tolerance = *sum_h;
    return;

}

