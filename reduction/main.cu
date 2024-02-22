#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include "reduction_baseline.h"
#include "reduction.h"



int
main(int argc, char *argv[]) {

    int   n;
    double  *res_h,*res_d;
    double  *a_h,*a_d;
    double start, stop_baseline, stop;

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

    // Blocks and threads
    dim3 dimBlock(256);
    dim3 dimGrid((n+dimBlock.x-1)/dimBlock.x);

    // Device warm-up
    reduction_baseline<<<dimGrid, dimBlock>>>(a_d, n, res_d);

    // Copy initializd array to devices
    cudaMemcpy(a_d, a_h, n * sizeof(double), cudaMemcpyHostToDevice);

    // Run baseline on device
    start = omp_get_wtime();
    cudaMemcpy(res_d, res_h, sizeof(double), cudaMemcpyHostToDevice);
    reduction_baseline<<<dimGrid, dimBlock>>>(a_d, n, res_d);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(res_h, res_d, sizeof(double), cudaMemcpyDeviceToHost);  
    stop_baseline = omp_get_wtime() - start;

    printf("\nreduction_baseline\n");
    printf("time: %.4f\n", stop_baseline);
    printf("result: %.4e\n", *res_h);

    // Initialize
    *res_h = 0.0;

    // Run reduction on device
    start = omp_get_wtime();
    cudaMemcpy(res_d, res_h, sizeof(double), cudaMemcpyHostToDevice);
    reduction<<<dimGrid, dimBlock>>>(a_d, n, res_d);
    cudaDeviceSynchronize();

    // Copy result back to host 
    cudaMemcpy(res_h, res_d, sizeof(double), cudaMemcpyDeviceToHost);  
    stop = omp_get_wtime() - start;

    printf("\nreduction\n");
    printf("time: %.4f\n", stop);
    printf("result: %.4e\n\n", *res_h);

    printf("speed-up: %.4f\n\n", stop_baseline/stop);

    // Clean-up
    cudaFreeHost(a_h); cudaFree(a_d); cudaFreeHost(res_h); cudaFree(res_d);

    return(0);
}
