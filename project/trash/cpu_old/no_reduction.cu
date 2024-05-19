#include <math.h>
#include <stdio.h>
#include <omp.h>

void
jacobi(double *** u, double *** uold, double *** f, int N, int iter_max, double* tolerance, int *n) {
    
    double delta = 2.0/(N+1), delta2 = delta*delta, frac = 1.0/6.0;
    while (*n < iter_max) {
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < N+1; i++) {
            for (int j = 1; j < N+1; j++) {
                for (int k = 1; k < N+1; k++) {
                    // Do iteration
                    u[i][j][k] = frac*(uold[i-1][j][k] + uold[i+1][j][k] + uold[i][j-1][k] + uold[i][j+1][k] + uold[i][j][k-1] + uold[i][j][k+1] + delta2*f[i][j][k]);
                }
            }
        }
        // Swap addresses
        double ***tmp;
        tmp = u;
        u = uold;
        uold = tmp;
        // Next iteration
        (*n)++;
    }

    return;

}
