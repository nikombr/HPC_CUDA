
#include <stdio.h>
#include <stdlib.h>


__device__ double reduction_step3(double *array, int n, int idx) {
    
    double reg = idx < n ? array[idx] : 0; // Make sure all the threads have indeed a value - should result in less error
    for (int dist = 16; dist > 0; dist /= 2)
        reg += __shfl_down_sync(-1, reg, dist);

    return reg;

}

__global__ void reduction_asyn(double *a, int n, int length, double *res) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    double var = 0;

    for (int i = idx; i < length; i += blockDim.x * gridDim.x) {
        var += a[i];
    }

    for (int dist = 16; dist > 0; dist /= 2)
        var += __shfl_down_sync(-1, var, dist);

    //double reg = reduction_step2(var, n, idx);

    // Allocate shared memory statically
    __shared__ double smem[32];

    if (threadIdx.x % 32 == 0) smem[threadIdx.x/32] = var;
    __syncthreads();

    idx = threadIdx.x;
    double reg = reduction_step3(smem, 32, idx);

    if (threadIdx.x == 0) atomicAdd(res, reg);
}

