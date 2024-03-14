
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

__host__ __device__
double map_pi_(double x, int n) {
    x = (x + 0.5) / n;
    return 4.0 / (n * (1.0 + x * x));
}
__host__ __device__
    double map_softmax_(double x, int n) {
    return exp((x + 1.0) / n);
}

#define map map_pi_


__host__ void reduction_openmp(double *a, int n, double *res) {

    double result = 0;

    #pragma omp parallel for reduction(+:result)
    for (int i = 0; i < n; i++) {
        result += map(a[i],n);
    }
    
    res[0] = result;
    
}

