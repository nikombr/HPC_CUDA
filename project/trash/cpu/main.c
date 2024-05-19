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

#include <omp.h>

int
main(int argc, char *argv[]) {

    if (argc < 5 || argc > 7) {
        printf("Usage: %s N(int) iter_max(int) tolerance(double) start_T(double) [output_type(0, 3 or 4)] [1 or 2]\n",argv[0]);
        return(1);
    }

    int     N;
    int 	iter_max = 1000;
    double	tolerance;
    double	start_T;
    int		output_type = 0;
    char	*output_prefix = "poisson_cpu";
    char    *output_ext = "";
    char    *extra_str = "";
    char	output_filename[FILENAME_MAX];
    double 	***u = NULL;
    double 	***uold = NULL;
    double 	***f = NULL;
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
        } else {
            extra_str = "_no_reduction";  // ouput type
        }

    }
    
    // Allocate memory
    if ( (u = malloc_3d(N+2, N+2, N+2)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }
    if ( (f = malloc_3d(N+2, N+2, N+2)) == NULL ) {
        perror("array f: allocation failed");
        exit(-1);
    }
    if ( (uold = malloc_3d(N+2, N+2, N+2)) == NULL ) {
        perror("array uold: allocation failed");
        exit(-1);
    }

    // Initialize and start and boundary conditions
    init(u, uold, f, N, start_T);

    // Call Jacobi iteration
    double start = omp_get_wtime();
    jacobi(u, uold, f, N, iter_max, &tolerance, &n);
    double stop = omp_get_wtime() - start;
    printf("%d %d %.5f %.5e # N iterations time error\n", N, n, stop, tolerance);
  
    // Dump  results if wanted 
    switch(output_type) {
	case 0:
	    // No output at all
	    break;
	case 3:
	    output_ext = ".bin";
	    sprintf(output_filename, "results/%s_%d%s%s", output_prefix, N,extra_str, output_ext);
	    fprintf(stderr, "\nWrite binary dump to %s\n", output_filename);
	    print_binary(output_filename, N, uold);
	    break;
	case 4:
	    output_ext = ".vtk";
	    sprintf(output_filename, "results/%s_%d%s", output_prefix, N, output_ext);
	    fprintf(stderr, "\nWrite VTK file to %s\n", output_filename);
	    print_vtk(output_filename, N, uold);
	    break;
	default:
	    fprintf(stderr, "Non-supported output type!\n");
	    break;
    }

    // De-allocate memory
    free_3d(u); free_3d(f); free_3d(uold);

    return(0);
}
