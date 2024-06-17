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
    *sum_h = tolerance + 1;
    
    while (n < iter_max && *sum_h > tolerance) {
        //*sum_h = 0.0;
        //cudaMemcpy(sum_d, sum_h, sizeof(double), cudaMemcpyHostToDevice);
 
        // Do iteration
        iteration(u_d, uold_d, f_d, N, sum_d);

        cudaMemcpy(sum_h, sum_d, sizeof(double), cudaMemcpyDeviceToHost);
        
        // Swap addresses
        swapArrays();
        // Next iteration
        n++;
    }
    tolerance = *sum_h;
    return;

}

