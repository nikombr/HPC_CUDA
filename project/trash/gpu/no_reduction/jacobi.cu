#include <math.h>
#include <stdio.h>
#include <omp.h>

__global__ void iteration(double *** u, double *** uold, double *** f, int N) {
    double delta = 2.0/(N+1), delta2 = delta*delta, frac = 1.0/6.0;
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 1;
    if ((i < N + 1) && (j < N + 1) && (k < N + 1)) {
        u[i][j][k] = frac*(uold[i-1][j][k] + uold[i+1][j][k] + uold[i][j-1][k] + uold[i][j+1][k] \
                        + uold[i][j][k+1] + uold[i][j][k-1] + delta2*f[i][j][k]);
    }
}

void
jacobi(double *** u, double *** uold, double *** f, int N, int iter_max, double* tolerance, int *n) {
   
    // Blocks and threads
    dim3 dimBlock(32,8,4);
    dim3 dimGrid(((N+2)+dimBlock.x-1)/dimBlock.x,((N+2)+dimBlock.y-1)/dimBlock.y,((N+2)+dimBlock.z-1)/dimBlock.z);
    //dim3 dimGrid(1,1,1);

    while (*n < iter_max) {
        
        // Do iteration
        iteration<<<dimGrid, dimBlock>>>(u, uold, f, N);
        cudaDeviceSynchronize();

        // Swap addresses
        double ***tmp;
        tmp = u;
        u = uold;
        uold = tmp;
        // Next iteration
        (*n)++;
       
    }
    return;

}
