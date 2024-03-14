#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include "reduction.h"
#include "reduction_openmp.h"

int
main(int argc, char *argv[]) {

    int   n;
    double  *res_h,*res_d;
    double  *a_h,*a_d;
    double start, stop_openmp, stop_gpu;

    n = 320;

    // command line argument sets the dimensions of the image
    if ( argc == 2 ) n = atoi(argv[1]);

    // Alloc memory on host and device
    cudaMallocHost(&a_h, n * sizeof(double));
    cudaMalloc(&a_d, n * sizeof(double));
    cudaMallocHost(&res_h, sizeof(double));
    cudaMalloc(&res_d, sizeof(double));
    if (a_h == NULL || a_d == NULL || res_h == NULL || res_d == NULL) {
       fprintf(stderr, "memory allocation failed!\n");
       return(1);
    }

    for (int i = 0; i < n; i++) {
        a_h[i] = i + 1.0;
    }

    // Initialize
    *res_h = 0.0;

    // Run reduction on host
    start = omp_get_wtime();
    reduction_openmp(a_h, n, res_h);
    stop_openmp = omp_get_wtime() - start;

    printf("\nreduction_openmp\n");
    printf("time: %.4f\n", stop_openmp);
    printf("result: %.4e\n\n", *res_h);

    // Blocks and threads
    dim3 dimBlock(256);
    dim3 dimGrid((n+dimBlock.x-1)/dimBlock.x);
    dimGrid.x = ceil((double)dimGrid.x/8); // 8 times less number of threads

    // Device warm-up
    reduction<<<dimGrid, dimBlock>>>(a_d, n, res_d);

    // Initialize
    *res_h = 0.0;

    // Copy initializd array to devices
    start = omp_get_wtime();
    cudaMemcpy(a_d, a_h, n * sizeof(double), cudaMemcpyHostToDevice);

    // Run reduction on device
    cudaMemcpy(res_d, res_h, sizeof(double), cudaMemcpyHostToDevice);
    reduction<<<dimGrid, dimBlock>>>(a_d, n, res_d);
    cudaDeviceSynchronize();

    // Copy result back to host 
    cudaMemcpy(res_h, res_d, sizeof(double), cudaMemcpyDeviceToHost);  
    stop_gpu = omp_get_wtime() - start;

    printf("\nreduction_gpu\n");
    printf("time: %.4f\n", stop_gpu);
    printf("result: %.4e\n\n", *res_h);

    // Clean-up
    cudaFreeHost(a_h); cudaFree(a_d); cudaFreeHost(res_h); cudaFree(res_d);

    return(0);
}
