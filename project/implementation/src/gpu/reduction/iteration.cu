#include <math.h>
#include <stdio.h>
#include <omp.h>

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
    
    // Index of thread
    int idx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    // Reduction
    double reg = val*val;
    for (int dist = 16; dist > 0; dist /= 2)
        reg += __shfl_down_sync(-1, reg, dist);
    __syncthreads();

    // Allocate shared memory statically
    __shared__ double smem[32];

    // Copy to shared memory
    if (idx % 32 == 0) smem[idx/32] = reg;
    __syncthreads();

    // Add the elements in shared memory together
    reg = idx < 32 ? smem[idx] : 0;
    for (int dist = 16; dist > 0; dist /= 2)
        reg += __shfl_down_sync(-1, reg, dist);

    if (idx == 0) atomicAdd(res, reg);
    
}

void iteration(double *** u, double *** uold, double *** f, int N, int iter_max, double *sum) {

    // Blocks and threads
    dim3 dimBlock(32,8,4);
    dim3 dimGrid(((N+2)+dimBlock.x-1)/dimBlock.x,((N+2)+dimBlock.y-1)/dimBlock.y,((N+2)+dimBlock.z-1)/dimBlock.z);

    // Do iteration
    iteration_inner<<<dimGrid, dimBlock>>>(u, uold, f, N, iter_max, sum);
    cudaDeviceSynchronize();
}