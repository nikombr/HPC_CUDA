#include <math.h>
#include <stdio.h>
#include <omp.h>

void
jacobi(double *** u, double *** uold, double *** f, int N, int iter_max, double* tolerance) {
    
    double delta = 2.0/(N+1), delta2 = delta*delta, frac = 1.0/6.0;
    double val, sum = *tolerance + 1;
    int n = 0;
    double start = omp_get_wtime();
    while (n < iter_max && sum > *tolerance) {
        sum = 0.0;
        #pragma omp parallel for private(val) reduction(+:sum) schedule(static)
        for (int i = 1; i < N+1; i++) {
            for (int j = 1; j < N+1; j++) {
                for (int k = 1; k < N+1; k++) {
                    // Do iteration
                    u[i][j][k] = frac*(uold[i-1][j][k] + uold[i+1][j][k] + uold[i][j-1][k] + uold[i][j+1][k] + uold[i][j][k-1] + uold[i][j][k+1] + delta2*f[i][j][k]);
                    // Check convergence with Frobenius norm
                    val = u[i][j][k] - uold[i][j][k];
                    sum += val*val;
                }
            }
        }
        // Swap addresses
        double ***tmp;
        tmp = u;
        u = uold;
        uold = tmp;
        // Next iteration
        n++;
    }
    double stop = omp_get_wtime() - start;
    *tolerance = sum;
    printf("%d %d %.5e %.5f # N iterations error time\n", N, n, *tolerance, stop);
    return;

}
