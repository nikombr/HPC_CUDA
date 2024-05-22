#include <stdio.h>
#include <stdlib.h>
//#include "../../lib/info_struct.h"
//#include "../../lib/poisson.h"

__global__ void iteration_inner(double *** u, double *** uold, double *** f, int N, int iter_max, int width) {
    double delta = 2.0/(N+1), delta2 = delta*delta, frac = 1.0/6.0;
    int k = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int i = threadIdx.z + blockIdx.z * blockDim.z + 1;
    if ((i < width + 1) && (j < N + 1) && (k < N + 1)) {
        u[i][j][k] = frac*(uold[i-1][j][k] + uold[i+1][j][k] + uold[i][j-1][k] + uold[i][j+1][k] + \
                           uold[i][j][k+1] + uold[i][j][k-1] + delta2*f[i][j][k]);

    }
}

void iteration(double *** u, double *** uold, double *** f, int N, int iter_max, int width) {
    // Blocks and threads
    dim3 dimBlock(32,8,4);
    dim3 dimGrid(((N+2)+dimBlock.x-1)/dimBlock.x,((N+2)+dimBlock.y-1)/dimBlock.y,((width+2)+dimBlock.z-1)/dimBlock.z);
    //dim3 dimGrid(1,1,1);
    iteration_inner<<<dimGrid, dimBlock>>>(u, uold, f, N, iter_max, width);
    cudaDeviceSynchronize();
}