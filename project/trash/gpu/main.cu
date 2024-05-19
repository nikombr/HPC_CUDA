#include <stdio.h>
#include <stdlib.h>
#include "utils/alloc3d.h"
#include "utils/print.h"
#include "utils/init.h"

#ifdef _NO_REDUCTION
#include "no_reduction/jacobi.h"
#endif

#ifdef _REDUCTION
#include "reduction/jacobi.h"
#endif

#ifdef _REDUCTION_ATOMIC
#include "reduction_atomic/jacobi.h"
#endif


#include <omp.h>

int
main(int argc, char *argv[]) {

    // Command line intpu
    if (argc < 5 || argc > 7) {
        printf("Usage: %s N(int) iter_max(int) tolerance(double) start_T(double) [output_type(0, 3 or 4)]\n",argv[0]);
        return(1);
    }

    int     N;
    int 	iter_max = 1000;
    double	tolerance;
    double	start_T;
    int		output_type = 0;
    char	*output_prefix = "poisson_gpu";
    char    *extra_str = "";
    char    *output_ext    = "";
    char	output_filename[FILENAME_MAX];
    double 	***u_h, ***u_d, ***uold_h, ***uold_d, ***f_h, ***f_d;
    //double *u_h, *u_d, *uold_h, *uold_d, *f_h, *f_d;
    double  *u_log, *uold_log, *f_log;
    int     n = 0;

    // Get the parameters from the command line
    N         = atoi(argv[1]);	// grid size
    iter_max  = atoi(argv[2]);  // max. no. of iterations
    tolerance = atof(argv[3]);  // tolerance
    start_T   = atof(argv[4]);  // start T for all inner grid points
    if (argc >= 6) {
	    output_type = atoi(argv[5]);  // ouput type
    }
    if (argc == 7) {
        if (atoi(argv[6]) == 1) {
	        extra_str = "_reduction";  // ouput type
        } else if (atoi(argv[6]) == 2) {
            extra_str = "_no_reduction";  // ouput type
        } else {
            extra_str = "_reduction_atomic";  // ouput type
        }

    }

    // Allocate memory
    u_h     = host_malloc_3d(N+2, N+2, N+2);
    uold_h  = host_malloc_3d(N+2, N+2, N+2);
    f_h     = host_malloc_3d(N+2, N+2, N+2);
    device_malloc_3d(&u_d,&u_log, N+2, N+2, N+2);
    device_malloc_3d(&uold_d, &uold_log, N+2, N+2, N+2);
    device_malloc_3d(&f_d, &f_log, N+2, N+2, N+2);
    /*cudaMalloc(&u_d,(N+2)*(N+2)*(N+2)*sizeof(double));
    cudaMalloc(&uold_d,(N+2)*(N+2)*(N+2)*sizeof(double));
    cudaMalloc(&f_d,(N+2)*(N+2)*(N+2)*sizeof(double));
    cudaMallocHost(&u_h,(N+2)*(N+2)*(N+2)*sizeof(double));
    cudaMallocHost(&uold_h,(N+2)*(N+2)*(N+2)*sizeof(double));
    cudaMallocHost(&f_h,(N+2)*(N+2)*(N+2)*sizeof(double));*/
    
    // Check allocation
    if (u_h == NULL || uold_h == NULL || f_h == NULL || u_d == NULL || uold_d == NULL || f_d == NULL) {
        perror("allocation failed");
        exit(-1);
    }


    // Initialize start and boundary conditions on host
    init(u_h, uold_h, f_h, N, start_T);

    // Do GPU warm-up
    //jacobi(u_d, uold_d, f_d, N, iter_max, &tolerance, &n);

    double start_transfer = omp_get_wtime();
    // Copy initializd array to devices
    cudaMemcpy(uold_log, **uold_h, (N+2) * (N+2) * (N+2) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(u_log, **u_h, (N+2) * (N+2) * (N+2) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(f_log, **f_h, (N+2) * (N+2) * (N+2) * sizeof(double), cudaMemcpyHostToDevice);

    // Call Jacobi iteration
    double start = omp_get_wtime();
    jacobi(u_d, uold_d, f_d, N, iter_max, &tolerance, &n);
    double stop = omp_get_wtime() - start;
    

    cudaMemcpy(**uold_h, uold_log, (N+2) * (N+2) * (N+2) * sizeof(double), cudaMemcpyDeviceToHost);
    double stop_transfer = omp_get_wtime() - start_transfer;
    printf("%d %d %.5f %.5f %.5e # N iterations time transfer_time error\n", N, n, stop, stop_transfer, tolerance);

    // Dump  results if wanted 
    switch(output_type) {
	case 0:
	    // No output at all
	    break;
	case 3:
	    output_ext = ".bin";
	    sprintf(output_filename, "results/%s_%d%s%s", output_prefix, N,extra_str, output_ext);
	    fprintf(stderr, "\nWrite binary dump to %s\n", output_filename);
	    print_binary(output_filename, N, uold_h);
	    break;
	case 4:
	    output_ext = ".vtk";
	    sprintf(output_filename, "results/%s_%d%s", output_prefix, N, output_ext);
	    fprintf(stderr, "\nWrite VTK file to %s\n", output_filename);
	    //print_vtk(output_filename, N, u_h);
	    break;
	default:
	    fprintf(stderr, "Non-supported output type!\n");
	    break;
    }

    // De-allocate memory
    host_free_3d(u_h); host_free_3d(f_h); host_free_3d(uold_h);
    device_free_3d(u_d, u_log); device_free_3d(f_d,f_log); device_free_3d(uold_d,uold_log);
    //cudaFree(u_d); cudaFree(uold_d); cudaFree(f_d);
    //cudaFreeHost(u_h); cudaFreeHost(uold_h); cudaFreeHost(f_h);

    return(0);
}
