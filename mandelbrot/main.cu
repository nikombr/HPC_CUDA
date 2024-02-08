#include <stdio.h>
#include <stdlib.h>
#include "mandel.h"
#include "writepng.h"
#ifdef _OPENMP
#include <omp.h>
#endif



int
main(int argc, char *argv[]) {

    int   width, height;
    int	  max_iter;
    int *image_h,*image_d;

    width    = 4800*2;
    height   = 4800*2;
    max_iter = 4000;

    // command line argument sets the dimensions of the image
    if ( argc == 2 ) width = height = atoi(argv[1]);

    // Alloc memory on host and device
    cudaMallocHost(&image_h, width * height * sizeof(int));
    cudaMalloc(&image_d, width * height * sizeof(int));
    if ( image_h == NULL || image_d == NULL) {
       fprintf(stderr, "memory allocation failed!\n");
       return(1);
    }

    double start, stop;

    // Device warm-up
    mandel<<<ceil((double)width/32),32>>>(width, height, image_d, max_iter);

    // Run on device
    start = omp_get_wtime();
    mandel<<<ceil((double)width/32),32>>>(width, height, image_d, max_iter);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(image_h, image_d, width * height * sizeof(int), cudaMemcpyDeviceToHost);  
    stop = omp_get_wtime() - start;

    printf("Device 1d time: %.4f\n", stop);

    // Illustrate in file
    writepng("mandelbrot_gpu_1d.png", image_h, width, height);

    // Run on device 2d (2 times for device warm-up)
    dim3 dimBlock(32,32,1);
    dim3 dimGrid(ceil((double)width/dimBlock.x),ceil((double)height/dimBlock.y),1);
    start = omp_get_wtime();
    mandel_2d<<<dimGrid,dimBlock>>>(width, height, image_d, max_iter);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(image_h, image_d, width * height * sizeof(int), cudaMemcpyDeviceToHost);  
    stop = omp_get_wtime() - start;
    printf("Device 2d time: %.4f\n", stop);

    // Illustrate in file
    writepng("mandelbrot_gpu_2d.png", image_h, width, height);

    // Run on host
    start = omp_get_wtime();
    mandel_omp(width, height, image_h, max_iter);
    stop = omp_get_wtime() - start;
    printf("Host time: %.4f\n", stop);

    // Illustrate in file
    writepng("mandelbrot_cpu.png", image_h, width, height);

    // Clean-up
    cudaFreeHost(image_h); cudaFree(image_d);

    return(0);
}
