#include <stdio.h>
#include <stdlib.h>
#include "../../lib/cudaMalloc3d.h"
#include "../../lib/init.h"
#include "../../lib/jacobi.h"
#include "../../lib/dumpOutput.h"
#include <omp.h>

void call(int N, int output_type, char *output_prefix, char*output_ext, char*extra_str, \
                        double tolerance, int iter_max, double start_T) {

    // Initialize
    double 	***u_h, ***u_d, ***uold_h, ***uold_d, ***f_h, ***f_d;
    double  *u_log, *uold_log, *f_log;
    int n = 0;

    // Allocation
    u_h     = host_malloc_3d(N+2, N+2, N+2);
    uold_h  = host_malloc_3d(N+2, N+2, N+2);
    f_h     = host_malloc_3d(N+2, N+2, N+2);
    device_malloc_3d(&u_d, &u_log, N+2, N+2, N+2);
    device_malloc_3d(&uold_d, &uold_log, N+2, N+2, N+2);
    device_malloc_3d(&f_d, &f_log, N+2, N+2, N+2);

    // Check allocation
    if (u_h == NULL || uold_h == NULL || f_h == NULL || u_d == NULL || uold_d == NULL || f_d == NULL) {
        perror("Allocation failed!");
        exit(-1);
    }

    // Initialize start and boundary conditions on host
    init(u_h, uold_h, f_h, N, start_T, 0, N + 1);

    // Do GPU warm-up
    jacobi(u_d, uold_d, f_d, N, iter_max, &tolerance, &n);

    double start_transfer = omp_get_wtime();
    // Copy initializd array to devices
    cudaMemcpy(uold_log, **uold_h, (N+2) * (N+2) * (N+2) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(   u_log,    **u_h, (N+2) * (N+2) * (N+2) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(   f_log,    **f_h, (N+2) * (N+2) * (N+2) * sizeof(double), cudaMemcpyHostToDevice);

    // Call Jacobi iteration
    double start = omp_get_wtime();
    jacobi(u_d, uold_d, f_d, N, iter_max, &tolerance, &n);
    double stop = omp_get_wtime() - start;

    cudaMemcpy(**uold_h, uold_log, (N+2) * (N+2) * (N+2) * sizeof(double), cudaMemcpyDeviceToHost);
    double stop_transfer = omp_get_wtime() - start_transfer;
    printf("%d %d %.5f %.5f %.5e # N iterations time transfer_time error\n", N, n, stop, stop_transfer, tolerance);

    // Dump  results if wanted
    output_prefix = "poisson_gpu"; 
    dump_output(output_type, output_prefix, output_ext, extra_str, N, uold_h);

    // De-allocate memory
    host_free_3d(u_h); host_free_3d(f_h); host_free_3d(uold_h);
    device_free_3d(u_d, u_log); device_free_3d(f_d,f_log); device_free_3d(uold_d,uold_log);
}