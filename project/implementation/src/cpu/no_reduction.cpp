#include <math.h>
#include <stdio.h>
#include <omp.h>
#include "../../lib/poisson.h"

void Poisson::jacobi() {
    
    double delta = 2.0/(N+1), delta2 = delta*delta, frac = 1.0/6.0;
    while (n < iter_max) {
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < N+1; i++) {
            for (int j = 1; j < N+1; j++) {
                for (int k = 1; k < N+1; k++) {
                    // Do iteration
                    u_h[i][j][k] = frac*(uold_h[i-1][j][k] + uold_h[i+1][j][k] + \
                                         uold_h[i][j-1][k] + uold_h[i][j+1][k] + \
                                         uold_h[i][j][k-1] + uold_h[i][j][k+1] + \
                                        delta2*f_h[i][j][k]);
                }
            }
        }
        // Swap addresses
        swapArrays();
        // Next iteration
        n++;
    }

    return;

}
