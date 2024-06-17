#include <stdio.h>
#include <stdlib.h>
#include "../../lib/malloc3d.h"
#include "../../lib/init.h"
#include "../../lib/jacobi.h"
#include "../../lib/dumpOutput.h"
#include "../../lib/poisson.h"
#include <omp.h>

void call(int N, int output_type, char *output_prefix, char*output_ext, char*extra_str, \
                        double tolerance, int iter_max, double start_T) {

    // Initialize
    Poisson poisson = Poisson(N, false, start_T, iter_max, tolerance);

    // Setup f matrix
    poisson.setup_f_matrix();

    // Allocation
    poisson.alloc();

    // Initialize matrices on host
    poisson.init();

    // Run Jacobi iterations
    double start = omp_get_wtime();
    poisson.jacobi();
    double stop = omp_get_wtime() - start;

    // Output to command line
    printf("%d %d %.5e %.5e # N iterations time error\n", poisson.N, poisson.n, stop, poisson.tolerance);

    // Finalize 
    poisson.finalize(output_type, output_ext, extra_str);
}