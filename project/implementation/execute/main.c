#include <stdio.h>
#include <stdlib.h>
#include "../lib/call.h"

int main(int argc, char *argv[]) {

    // Initialize
    int     N;                              // Dimensions
    int 	iter_max;                       // Maximum number of iterations in Jacobi
    double	tolerance;                      // Tolerance, only needed if convergence test is executed
    double	start_T;                        // Initial value in the interior of u and uold
    int		output_type = 0;                // Default output type = no outpu
    char	*output_prefix = "poisson";     // Prefix for output file
    char    *extra_str = "";                // Added string for overview
    char    *output_ext    = "";            // Output type
    int     n = 0;                          // Iterator

    // Command line input, check correct usage
    if (argc < 5 || argc > 7) {
        printf("Usage: %s N(int) iter_max(int) tolerance(double) start_T(double) \
                [output_type(0, 3 or 4)] [extra_str(1, 2 or 3)]\n",argv[0]);
        return(1);
    }

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
	        extra_str = "_reduction"; 
        } else if (atoi(argv[6]) == 2) {
            extra_str = "_no_reduction";  // ouput type
        } else {
            extra_str = "_reduction_atomic";  // ouput type
        }
    }

    // Allocate memory
    #ifdef _CUDA
        output_prefix = "poisson_gpu"
        double 	***u_h, ***u_d, ***uold_h, ***uold_d, ***f_h, ***f_d;
        double  *u_log, *uold_log, *f_log;
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
    #endif

    // Call all the functions needed to compute solution
    call(N, output_type, output_prefix, output_ext, extra_str, tolerance, iter_max, start_T);
    
    /*// Initialize start and boundary conditions on host
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
    dump_output(output_type, output_prefix, output_ext, output_filename, extra_str, N, u);

    // De-allocate memory
    host_free_3d(u_h); host_free_3d(f_h); host_free_3d(uold_h);
    device_free_3d(u_d, u_log); device_free_3d(f_d,f_log); device_free_3d(uold_d,uold_log);*/

    return(0);
}
