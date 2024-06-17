#include <math.h>
#include <stdio.h>
#include <omp.h>
#include "../../lib/poisson.h"

void Poisson::jacobi() {
    
    double delta = 2.0/(this->N+1), delta2 = delta*delta, frac = 1.0/6.0;
    while (this->n < this->iter_max) {
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < this->N+1; i++) {
            for (int j = 1; j < this->N+1; j++) {
                for (int k = 1; k < this->N+1; k++) {
                    // Do iteration
                    this->u_h[i][j][k] = frac*(this->uold_h[i-1][j][k] + this->uold_h[i+1][j][k] + \
                                               this->uold_h[i][j-1][k] + this->uold_h[i][j+1][k] + \
                                               this->uold_h[i][j][k-1] + this->uold_h[i][j][k+1] + \
                                               delta2*this->f_h[i][j][k]);
                }
            }
        }
        // Swap addresses
        this->swapArrays();
        // Next iteration
        (this->n)++;
    }

    return;

}
