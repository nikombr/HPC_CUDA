#include <math.h>
#include <stdio.h>
#include <omp.h>
#include "../../../lib/poisson.h"
#include "../../../lib/iteration.h"
#include <cuda_runtime_api.h>

void Poisson::jacobi() {

    while (n < iter_max) {
        
        // Do iteration
        iteration(deviceData[0].u_d, deviceData[0].uold_d, deviceData[0].f_d, N);

        // Swap addresses
        swapArrays();

        // Next iteration
        n++;
    }
    return;

}
