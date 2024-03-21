#include <math.h>
#include <stdio.h>
#include <omp.h>

__global__ void iteration(double *** u, double *** uold, double *** f, int N, int iter_max) {
    double delta = 2.0/(N+1), delta2 = delta*delta, frac = 1.0/6.0;
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 1;
    if (i < N + 1 & j < N + 1 & k < N + 1) {
        u[i][j][k] = frac*(uold[i-1][j][k] + uold[i+1][j][k] + uold[i][j-1][k] + uold[i][j+1][k] + uold[i][j][k-1] + uold[i][j][k+1] + delta2*f[i][j][k]);
    }
}

void
jacobi(double *** u, double *** uold, double *** f, int N, int iter_max, double* tolerance) {

    // Blocks and threads
    dim3 dimBlock(256,256,256);
    dim3 dimGrid((N+dimBlock.x-1)/dimBlock.x,(N+dimBlock.y-1)/dimBlock.y,(N+dimBlock.z-1)/dimBlock.z);

    int n = 0;
    double start = omp_get_wtime();
    while (n < iter_max) {
 
        // Do iteration
        iteration<<<dimGrid, dimBlock>>>(u, uold, f, N, iter_max);
        cudaDeviceSynchronize();
     
        // Swap addresses
        double ***tmp;
        tmp = u;
        u = uold;
        uold = tmp;
        // Next iteration
        n++;
    }
    double stop = omp_get_wtime() - start;
    printf("%d %d %.5f # N iter_max time\n", N, iter_max, stop);
    return;

}
