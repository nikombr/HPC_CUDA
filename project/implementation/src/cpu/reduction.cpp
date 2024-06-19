#include <math.h>
#include <stdio.h>
#include <omp.h>
#include "../../lib/poisson.h"

void Poisson::jacobi() {
    
    double delta = 2.0/(this->N+1), delta2 = delta*delta, frac = 1.0/6.0;
    double val, sum = this->tolerance + 1;
    while (n < iter_max && sum > tolerance) {
        sum = 0.0;
        #pragma omp parallel for private(val) reduction(+:sum) schedule(static)
        for (int i = 1; i < N+1; i++) {
            for (int j = 1; j < N+1; j++) {
                for (int k = 1; k < N+1; k++) {
                    // Do iteration
                    u_h[i][j][k] = frac*(uold_h[i-1][j][k] + uold_h[i+1][j][k] + \
                                               uold_h[i][j-1][k] + uold_h[i][j+1][k] + \
                                               uold_h[i][j][k-1] + uold_h[i][j][k+1] + \
                                               delta2*f_h[i][j][k]);
                    // Check convergence with Frobenius norm
                    val = u_h[i][j][k] - uold_h[i][j][k];
                    sum += val*val;
                }
            }
        }
        // Swap addresses
        swapArrays();
        // Next iteration
        (n)++;
    }
    tolerance = sum;
    return;

}
