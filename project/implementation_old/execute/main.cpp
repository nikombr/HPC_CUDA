#include <stdio.h>
#include <stdlib.h>
#ifdef _NORMAL_MAKE
    #include <mpi.h>
#endif
#include "../lib/call.h"
/*#ifdef _CPU_GPU
    #include "../lib/call.h"
#endif
#ifdef _MGPU
    #include "../lib/call2.h"
#endif*/

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
        } else if (atoi(argv[6]) == 3) {
            extra_str = "_reduction_atomic";  // ouput type
        } else if (atoi(argv[6]) == 4) {
            extra_str = "_no_reduction_asyn";  // ouput type
        }
    }
    call(N, output_type, output_prefix, output_ext, extra_str, tolerance, iter_max, start_T);

    return(0);
}
