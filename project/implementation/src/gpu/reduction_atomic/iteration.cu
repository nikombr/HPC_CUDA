#include <math.h>
#include <stdio.h>
#include <omp.h>
#include <cuda_runtime_api.h>

__global__ void iteration_inner(double *** u, double *** uold, double *** f, int N, int iter_max, double *res) {
    double val = 0;
    double delta = 2.0/(N+1), delta2 = delta*delta, frac = 1.0/6.0;
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 1;
    if ((i < N + 1) && (j < N + 1) && (k < N + 1)) {
        u[i][j][k] = frac*(uold[i-1][j][k] + uold[i+1][j][k] + uold[i][j-1][k] + uold[i][j+1][k] \
                        + uold[i][j][k+1] + uold[i][j][k-1] + delta2*f[i][j][k]);
        val = u[i][j][k] - uold[i][j][k];
    }

    double reg = val*val;
    atomicAdd(res,reg);
    
}

void iteration(double *** u, double *** uold, double *** f, int N, int iter_max, double *sum) {

    // Blocks and threads
    dim3 dimBlock(32,4,2);
    dim3 dimGrid(((N+1)+dimBlock.x-1)/dimBlock.x,((N+1)+dimBlock.y-1)/dimBlock.y,((N+1)+dimBlock.z-1)/dimBlock.z);

    // Do iteration
    iteration_inner<<<dimGrid, dimBlock>>>(u, uold, f, N, iter_max, sum);
    cudaDeviceSynchronize();
}