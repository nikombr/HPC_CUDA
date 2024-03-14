#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include "reduction_baseline.h"
#include "reduction.h"
#include "reduction_smem.h"
#include "reduction_presum.h"
#include "reduction_asyn.h"
#include "reduction_openmp.h"


void asyn(cudaStream_t* stream,double*res_h,double*res_d,double*a_d,double*a_h,int n,int SPLITS,double*stop ) {
    double start = omp_get_wtime();
    
    for (int s = 0; s < SPLITS; ++s)
        cudaStreamCreate(&stream[s]);
    

    // Allocate array for storing temporal sum and initilize to zero
    double *resv_d, *resv_h;
    cudaMallocHost(&resv_h, SPLITS * sizeof(double));
    cudaMalloc(&resv_d, SPLITS * sizeof(double));
    cudaMemset(resv_d, 0, SPLITS * sizeof(double));

    // Blocks and threads
    dim3 dimBlock(256);
    dim3 dimGrid((n+dimBlock.x-1)/dimBlock.x);
    dimGrid.x = ceil((double)dimGrid.x/8); // 8 times less number of threads

    // Asyncrounously do the computation
    for (int s = 0; s < SPLITS; ++s) {
        int length = n / SPLITS;
        int offset = s * length;
        cudaMemcpyAsync(a_d + offset, a_h + offset, length * sizeof(double), cudaMemcpyHostToDevice, stream[s]);
        reduction_asyn<<<dimGrid.x / SPLITS, dimBlock, 0, stream[s]>>>(a_d + offset, n, length, resv_d + s);
    }
    cudaDeviceSynchronize();

    // Copy result back to host 
    cudaMemcpy(resv_h, resv_d, SPLITS*sizeof(double), cudaMemcpyDeviceToHost);
    for (int s = 0; s < SPLITS; ++s) *res_h += resv_h[s];  

    *stop = omp_get_wtime() - start;
}


int
main(int argc, char *argv[]) {

    int   n;
    double  *res_h,*res_d;
    double  *a_h,*a_d;
    double start, stop_baseline, stop, stop_smem, stop_presum,stop_presum_transfer, stop_openmp, stop_asyn;

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
    cudaMemcpy(res_d, res_h, sizeof(double), cudaMemcpyHostToDevice);
    start = omp_get_wtime();
    reduction_baseline<<<dimGrid, dimBlock>>>(a_d, n, res_d);
    cudaDeviceSynchronize();
    stop_baseline = omp_get_wtime() - start;

    // Copy result back to host
    cudaMemcpy(res_h, res_d, sizeof(double), cudaMemcpyDeviceToHost);  

    printf("\nreduction_baseline\n");
    printf("time: %.4f\n", stop_baseline);
    printf("result: %.4e\n", *res_h);

    // Initialize
    *res_h = 0.0;

    // Run reduction on device
    cudaMemcpy(res_d, res_h, sizeof(double), cudaMemcpyHostToDevice);
    start = omp_get_wtime();
    reduction<<<dimGrid, dimBlock>>>(a_d, n, res_d);
    cudaDeviceSynchronize();
    stop = omp_get_wtime() - start;

    // Copy result back to host 
    cudaMemcpy(res_h, res_d, sizeof(double), cudaMemcpyDeviceToHost);  

    printf("\nreduction\n");
    printf("time: %.4f\n", stop);
    printf("result: %.4e\n\n", *res_h);

    printf("speed-up: %.4f\n\n", stop_baseline/stop);
    

    // Initialize
    *res_h = 0.0;

    // Run reduction on device
    cudaMemcpy(res_d, res_h, sizeof(double), cudaMemcpyHostToDevice);
    start = omp_get_wtime();
    reduction_smem<<<dimGrid, dimBlock>>>(a_d, n, res_d);
    cudaDeviceSynchronize();
    stop_smem = omp_get_wtime() - start;

    // Copy result back to host 
    cudaMemcpy(res_h, res_d, sizeof(double), cudaMemcpyDeviceToHost);  

    printf("\nreduction_smem\n");
    printf("time: %.4f\n", stop_smem);
    printf("result: %.4e\n\n", *res_h);
    printf("speed-up vs baseline: %.4f\n\n", stop_baseline/stop_smem);
    printf("speed-up vs reduction warp: %.4f\n\n", stop/stop_smem);

    // Initialize
    *res_h = 0.0;

    // Blocks and threads
    dimGrid.x = ceil((double)dimGrid.x/8); // 8 times less number of threads
    printf("%d",dimGrid.x);

    // Run reduction on device
    cudaMemcpy(res_d, res_h, sizeof(double), cudaMemcpyHostToDevice);
    start = omp_get_wtime();
    reduction_presum<<<dimGrid, dimBlock>>>(a_d, n, res_d);
    cudaDeviceSynchronize();
    stop_presum = omp_get_wtime() - start;

    // Copy result back to host 
    cudaMemcpy(res_h, res_d, sizeof(double), cudaMemcpyDeviceToHost);  

    printf("\nreduction_presum\n");
    printf("time: %.4f\n", stop);
    printf("result: %.4e\n\n", *res_h);
    printf("speed-up vs baseline: %.4f\n\n", stop_baseline/stop_presum);
    printf("speed-up vs reduction warp: %.4f\n\n", stop/stop_presum);
    printf("speed-up vs shared memory: %.4f\n\n", stop_smem/stop_presum);

    // Initialize
    *res_h = 0.0;

    // Blocks and threads
    dimGrid.x = ceil((double)dimGrid.x/8); // 8 times less number of threads
    printf("%d",dimGrid.x);

    // Run reduction on device
    start = omp_get_wtime();
    // Copy initializd array to devices
    cudaMemcpy(a_d, a_h, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(res_d, res_h, sizeof(double), cudaMemcpyHostToDevice);
    reduction_presum<<<dimGrid, dimBlock>>>(a_d, n, res_d);
    cudaDeviceSynchronize();

    // Copy result back to host 
    cudaMemcpy(res_h, res_d, sizeof(double), cudaMemcpyDeviceToHost);  
    stop_presum_transfer = omp_get_wtime() - start;

    // Initialize
    *res_h = 0.0;


    // Setup streams
    #define SPLITS 4
    cudaStream_t stream[SPLITS];
    asyn(stream,res_h,res_d,a_d,a_h,n,SPLITS,&stop_asyn);

    printf("asyn result: %.4e\n\n", *res_h);

    // Initialize
    *res_h = 0.0;

    // Run reduction on host
    start = omp_get_wtime();
    reduction_openmp(a_h, n, res_h);
    stop_openmp = omp_get_wtime() - start;

    printf("\nreduction_openmp\n");
    printf("time: %.4f\n", stop_openmp);
    printf("result: %.4e\n\n", *res_h);

    printf("baseline:\t\t%.4f\n", stop_baseline);
    printf("warp-shuffles:\t\t%.4f\n", stop);
    printf("smem:\t\t\t%.4f\n", stop_smem);
    printf("presum:\t\t\t%.4f\n", stop_presum);
    printf("openmp (cpu):\t\t%.4f\n", stop_openmp);
    printf("presum + transfer:\t%.4f\n", stop_presum_transfer);
    printf("asyn:\t\t\t%.4f\n", stop_asyn);
    printf("speed-up:\t\t%.4f %%\n", (stop_presum_transfer-stop_asyn)/stop_presum_transfer*100);

    // Clean-up
    cudaFreeHost(a_h); cudaFree(a_d); cudaFreeHost(res_h); cudaFree(res_d);

    return(0);
}
