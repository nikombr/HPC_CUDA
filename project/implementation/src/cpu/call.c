#include <stdio.h>
#include <stdlib.h>
#include "../../lib/cudaMalloc3d.h"
#include "../../lib/malloc3d.h"
#include "../../lib/init.h"
#include "../../lib/jacobi.h"
#include "../../lib/dumpOutput.h"
#include <omp.h>

void call(int N, int output_type, char *output_prefix, char*output_ext, char*extra_str, \
                        double tolerance, int iter_max, double start_T) {

    // Initialize
    double ***u, ***uold, ***f;
    int n = 0;

    // Allocation
    u = malloc_3d(N+2, N+2, N+2);
    uold = malloc_3d(N+2, N+2, N+2);
    f = malloc_3d(N+2, N+2, N+2);
    
    // Check allocation
    if ( u == NULL ||  uold == NULL || f == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }

    // Initialize and start and boundary conditions
    init(u, uold, f, N, start_T);

    // Call Jacobi iteration
    double start = omp_get_wtime();
    jacobi(u, uold, f, N, iter_max, &tolerance, &n);
    double stop = omp_get_wtime() - start;

    // Output to command line
    printf("%d %d %.5f %.5e # N iterations time error\n", N, n, stop, tolerance);

    // Dump  results if wanted 
    dump_output(output_type, output_prefix, output_ext, extra_str, N, uold);
}