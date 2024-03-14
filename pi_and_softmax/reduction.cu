
#include <stdio.h>
#include <stdlib.h>


__host__ __device__
double map_pi(double x, int n) {
    x = (x + 0.5) / n;
    return 4.0 / (n * (1.0 + x * x));
}
__host__ __device__
    double map_softmax(double x, int n) {
    return exp((x + 1.0) / n);
}

#define map map_pi

__device__ double reduction_step(double *array, int n, int idx) {
    
    double reg = idx < n ? array[idx] : 0; // Make sure all the threads have indeed a value - should result in less error
    for (int dist = 16; dist > 0; dist /= 2)
        reg += __shfl_down_sync(-1, reg, dist);

    return reg;

}

__global__ void reduction(double *a, int n, double *res) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    double var = 0;

    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        var += map(a[i],n);
    }

    for (int dist = 16; dist > 0; dist /= 2)
        var += __shfl_down_sync(-1, var, dist);

    // Allocate shared memory statically
    __shared__ double smem[32];

    if (threadIdx.x % 32 == 0) smem[threadIdx.x/32] = var;
    __syncthreads();

    idx = threadIdx.x;
    double reg = reduction_step(smem, 32, idx);

    if (threadIdx.x == 0) atomicAdd(res, reg);
}

