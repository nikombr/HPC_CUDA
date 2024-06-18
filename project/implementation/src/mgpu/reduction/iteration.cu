#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
//#include "../../lib/info_struct.h"
//#include "../../lib/poisson.h"

__device__ void reduction(double val, double*res) {
    // Index of thread
    int idx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    // Reduction
    double reg = val*val;
    for (int dist = 16; dist > 0; dist /= 2)
        reg += __shfl_down_sync(-1, reg, dist);
    __syncthreads();
    // Allocate shared memory statically
    __shared__ double smem[32];
    memset(smem,0,32*sizeof(double));
    // Copy to shared memory
    if (idx % 32 == 0) smem[idx/32] = reg;
    __syncthreads();
    // Add the elements in shared memory together
    reg = idx < 32 ? smem[idx] : 0;
    for (int dist = 16; dist > 0; dist /= 2)
        reg += __shfl_down_sync(-1, reg, dist);
    if (idx == 0) atomicAdd(res, reg);
}

__global__ void iteration_inner(double *** u, double *** uold, double *** f, int N, int end, double*res, double delta2, double frac) {
    double val = 0.0;
    int k = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int i = threadIdx.z + blockIdx.z * blockDim.z + 1;
    if ((i < end + 1) && (j < N + 1) && (k < N + 1)) {
        u[i][j][k] = frac*(uold[i-1][j][k] + uold[i+1][j][k] + uold[i][j-1][k] + uold[i][j+1][k] \
                        + uold[i][j][k+1] + uold[i][j][k-1] + delta2*f[i][j][k]);
        val = u[i][j][k] - uold[i][j][k];
    }
    reduction(val, res);
}

__global__ void iteration_lower_boundary(double *** u, double *** uold, double *** uold_peer, double *** f, int N, int peer_width, double*res, double delta2, double frac) {
    double val = 0.0;
    int k = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    if ((j < N + 1) && (k < N + 1)) {
        u[i][j][k] = frac*(uold_peer[peer_width][j][k] + uold[i+1][j][k] + uold[i][j-1][k] + uold[i][j+1][k] \
                        + uold[i][j][k+1] + uold[i][j][k-1] + delta2*f[i][j][k]);
        val = u[i][j][k] - uold[i][j][k];
    }
    reduction(val, res);
}

__global__ void iteration_upper_boundary(double *** u, double *** uold, double *** uold_peer, double *** f, int N, int width, double*res, double delta2, double frac) {
    double val = 0.0;
    int k = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int i = threadIdx.z + blockIdx.z * blockDim.z + width;
    if ((j < N + 1) && (k < N + 1)) {
        u[i][j][k] = frac*(uold[i-1][j][k] + uold_peer[0][j][k] + uold[i][j-1][k] + uold[i][j+1][k] \
                        + uold[i][j][k+1] + uold[i][j][k-1] + delta2*f[i][j][k]);
        val = u[i][j][k] - uold[i][j][k];
    }
    reduction(val, res);
}

__global__ void init_zero(double *res) {
    *res = 0.0;
}


__global__ void add(double *res, double *res_bound) {
    *res += *res_bound;
}

void iteration(double *** u, double *** uold, double *** uold_peer, double *** f, int N,  double*sum, double*sum_bound, int width, int peer_width, int canAccessPeerPrev, int canAccessPeerNext,cudaStream_t  stream) {
    
    init_zero<<<1, 1, 0, stream>>>(sum);
    init_zero<<<1, 1, 0, stream>>>(sum_bound);
    cudaStreamSynchronize(stream);
    double delta = 2.0/(N+1), delta2 = delta*delta, frac = 1.0/6.0;
    int end = width - canAccessPeerNext - canAccessPeerPrev;
    // Blocks and threads
    dim3 dimBlock(32,4,2);
    dim3 dimBlockBound(dimBlock.x,dimBlock.y,1);
    dim3 dimGrid((N+dimBlock.x-1)/dimBlock.x,(N+dimBlock.y-1)/dimBlock.y,(end+dimBlock.z-1)/dimBlock.z);
    dim3 dimGridBound((N+dimBlock.x-1)/dimBlock.x,(N+dimBlock.y-1)/dimBlock.y,1);
    // Kernel calls
    if (canAccessPeerPrev) {
        iteration_lower_boundary<<<dimGridBound, dimBlockBound, 0, stream>>>(u, uold, uold_peer, f, N, peer_width, sum_bound, delta2, frac);
    }
    else if (canAccessPeerNext) { // In our setting the GPUs can only have peer-acces in one direction
        iteration_upper_boundary<<<dimGridBound, dimBlockBound, 0, stream>>>(u, uold, uold_peer, f, N, width, sum_bound, delta2, frac);
    }
    iteration_inner<<<dimGrid, dimBlock,0,stream>>>(u, uold, f, N, end, sum, delta2, frac);
    cudaStreamSynchronize(stream);
    add<<<1, 1,0, stream>>>(sum, sum_bound);
    //gpuErrchk( cudaPeekAtLastError() );
    //gpuErrchk( cudaDeviceSynchronize() );
    //cudaDeviceSynchronize();
}