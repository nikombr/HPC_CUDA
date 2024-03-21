#include <math.h>
#include <stdio.h>
#include <omp.h>

__global__ void iteration(double *** u, double *** uold, double *** f, int N, int iter_max) {
    double val, sum = *tolerance + 1;
    double delta = 2.0/(N+1), delta2 = delta*delta, frac = 1.0/6.0;
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 1;
    if (i < N + 1 & j < N + 1 & k < N + 1) {
        u[i][j][k] = frac*(uold[i-1][j][k] + uold[i+1][j][k] + uold[i][j-1][k] + uold[i][j+1][k] + uold[i][j][k-1] + uold[i][j][k+1] + delta2*f[i][j][k]);
        val = u[i][j][k] - uold[i][j][k];
        sum += val*val;
    }
}
/*__device__ double reduction_step3(double *array, int n, int idx) {
    
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
}*/

void
jacobi(double *** u, double *** uold, double *** f, int N, int iter_max, double* tolerance) {

    // Blocks and threads
    dim3 dimBlock(256,256,256);
    dim3 dimGrid((N+dimBlock.x-1)/dimBlock.x,(N+dimBlock.y-1)/dimBlock.y,(N+dimBlock.z-1)/dimBlock.z);

    int n = 0;
    double start = omp_get_wtime();
    while (n < iter_max && sum > *tolerance) {
 
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
    *tolerance = sum;
    printf("%d %d %.5f %.5e # N iterations time error\n", N, n, stop, *tolerance);
    return;

}

/*void
jacobi(double *** u, double *** uold, double *** f, int N, int iter_max, double* tolerance) {
    
    double delta = 2.0/(N+1), delta2 = delta*delta, frac = 1.0/6.0;
    double val, sum = *tolerance + 1;
    int n = 0;
    double start = omp_get_wtime();
    while (n < iter_max && sum > *tolerance) {
        sum = 0.0;
        
        for (int i = 1; i < N+1; i++) {
            for (int j = 1; j < N+1; j++) {
                for (int k = 1; k < N+1; k++) {
                    // Do iteration
                    u[i][j][k] = frac*(uold[i-1][j][k] + uold[i+1][j][k] + uold[i][j-1][k] + uold[i][j+1][k] + uold[i][j][k-1] + uold[i][j][k+1] + delta2*f[i][j][k]);
                    // Check convergence with Frobenius norm
                    val = u[i][j][k] - uold[i][j][k];
                    sum += val*val;
                }
            }
        }
        // Swap addresses
        double ***tmp;
        tmp = u;
        u = uold;
        uold = tmp;
        // Next iteration
        n++;
    }
    double stop = omp_get_wtime() - start;
    *tolerance = sum;
    printf("%d %d %.5f %.5e # N iterations time error\n", N, n, stop, *tolerance);
    return;

}
*/