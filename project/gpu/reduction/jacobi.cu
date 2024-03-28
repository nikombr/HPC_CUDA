#include <math.h>
#include <stdio.h>
#include <omp.h>

__global__ void iteration(double *** u, double *** uold, double *** f, int N, int iter_max, double *res) {
    double val = 0;
    double delta = 2.0/(N+1), delta2 = delta*delta, frac = 1.0/6.0;
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 1;
    double val1, val2;
    
    if ((i < N + 1) && (j < N + 1) && (k < N + 1)) {
        // Do iteration
        val2 = uold[i][j][k];
        val1 = frac*(uold[i-1][j][k] + uold[i+1][j][k] + uold[i][j-1][k] + uold[i][j+1][k] + uold[i][j][k-1] + uold[i][j][k+1] + delta2*f[i][j][k]);
        // Check convergence with Frobenious norm
        val = val1 - val2;
        double reg = val*val;
        atomicAdd(res,reg);
    }
    /*
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
    */
    
}


void
jacobi(double *** u, double *** uold, double *** f, int N, int iter_max, double* tolerance) {
    double  *sum_h, *sum_d;
    
    cudaMallocHost(&sum_h, sizeof(double));
    cudaMalloc(&sum_d, sizeof(double));
    *sum_h = *tolerance + 1;
    
    // Blocks and threads
    dim3 dimBlock(256,256,256);
    dim3 dimGrid((N+dimBlock.x-1)/dimBlock.x,(N+dimBlock.y-1)/dimBlock.y,(N+dimBlock.z-1)/dimBlock.z);
    int n = 0;
    double start = omp_get_wtime();
    while (n < iter_max && *sum_h > *tolerance) {
        *sum_h = 1.234;
        cudaMemcpy(sum_d, sum_h, sizeof(double), cudaMemcpyHostToDevice);
 
        // Do iteration
        iteration<<<dimGrid, dimBlock>>>(u, uold, f, N, iter_max, sum_d);
        cudaDeviceSynchronize();

        cudaMemcpy(sum_h, sum_d, sizeof(double), cudaMemcpyDeviceToHost);
        printf("%.5e, %.5e\n",*sum_h,*tolerance);
        // Swap addresses
        double ***tmp;
        tmp = u;
        u = uold;
        uold = tmp;
        // Next iteration
        n++;
    }
    double stop = omp_get_wtime() - start;
    *tolerance = *sum_h;
    printf("%d %d %.5f %.5e # N iterations time error\n", N, n, stop, *sum_h);
    return;

}

