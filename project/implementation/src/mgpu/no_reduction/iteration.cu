#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
__global__ void iteration_inner(double *** u, double *** uold, double *** f, int N, int end, double delta2, double frac) {
    int k = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int i = threadIdx.z + blockIdx.z * blockDim.z + 1;
    if ((i < end + 1) && (j < N + 1) && (k < N + 1)) {
        u[i][j][k] = frac*(uold[i-1][j][k] + uold[i+1][j][k] + uold[i][j-1][k] + uold[i][j+1][k] \
                        + uold[i][j][k+1] + uold[i][j][k-1] + delta2*f[i][j][k]);
    }
}

__global__ void iteration_lower_boundary(double *** u, double *** uold, double *** uold_peer, double *** f, int N, int peer_width, double delta2, double frac) {
    int k = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    if ((j < N + 1) && (k < N + 1)) {
        u[i][j][k] = frac*(uold_peer[peer_width][j][k] + uold[i+1][j][k] + uold[i][j-1][k] + uold[i][j+1][k] \
                        + uold[i][j][k+1] + uold[i][j][k-1] + delta2*f[i][j][k]);
        
    }
}

__global__ void iteration_upper_boundary(double *** u, double *** uold, double *** uold_peer, double *** f, int N, int width, double delta2, double frac) {
    int k = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int i = threadIdx.z + blockIdx.z * blockDim.z + width;
    if ((j < N + 1) && (k < N + 1)) {
        u[i][j][k] = frac*(uold[i-1][j][k] + uold_peer[0][j][k] + uold[i][j-1][k] + uold[i][j+1][k] \
                        + uold[i][j][k+1] + uold[i][j][k-1] + delta2*f[i][j][k]);
    }
}

void iteration(double *** u, double *** uold, double *** uold_peer, double *** f, int N, int width, int peer_width, int canAccessPeerPrev, int canAccessPeerNext) {
    
    double delta = 2.0/(N+1), delta2 = delta*delta, frac = 1.0/6.0;
    int end = width - canAccessPeerNext - canAccessPeerPrev;
    // Blocks and threads
    dim3 dimBlock(32,4,2);
    dim3 dimBlockBound(dimBlock.x,dimBlock.y,1);
    dim3 dimGrid((N+dimBlock.x-1)/dimBlock.x,(N+dimBlock.y-1)/dimBlock.y,(end+dimBlock.z-1)/dimBlock.z);
    dim3 dimGridBound((N+dimBlock.x-1)/dimBlock.x,(N+dimBlock.y-1)/dimBlock.y,1);
    // Kernel calls
    if (canAccessPeerPrev) {
        iteration_lower_boundary<<<dimGridBound, dimBlockBound>>>(u, uold, uold_peer, f, N, peer_width, delta2, frac);
    }
    else if (canAccessPeerNext) { // In our setting the GPUs can only have peer-acces in one direction
        iteration_upper_boundary<<<dimGridBound, dimBlockBound>>>(u, uold, uold_peer, f, N, width, delta2, frac);
    }
    iteration_inner<<<dimGrid, dimBlock>>>(u, uold, f, N, end, delta2, frac);
    //gpuErrchk( cudaPeekAtLastError() );
    //gpuErrchk( cudaDeviceSynchronize() );
    //cudaDeviceSynchronize();
}