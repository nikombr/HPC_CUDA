
#include <stdio.h>
#include <stdlib.h>

__global__ void reduction(double *a, int n, double *res) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        double reg = a[idx];

        for (int dist = 16; dist > 0; dist /= 2)
            reg += __shfl_down_sync(-1, reg, dist);

        if (threadIdx.x % 32 == 0) atomicAdd(res, reg);
    }
}
