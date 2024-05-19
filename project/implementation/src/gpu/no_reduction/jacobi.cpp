#include <math.h>
#include <stdio.h>
#include <omp.h>
#include "../../../lib/poisson.h"
#include "../../../lib/iteration.h"
#include <cuda_runtime_api.h>

void Poisson::jacobi() {

    while (this->n < this->iter_max) {
        
        // Do iteration
        iteration(this->u_d, this->uold_d, this->f_d, this->N, this->iter_max);

        // Swap addresses
        this->swapArrays();

        // Next iteration
        (this->n)++;
       
    }
    return;

}
