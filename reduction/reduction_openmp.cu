
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


__host__ void reduction_openmp(double *a, int n, double *res) {

    double result = 0;

    #pragma omp parallel for reduction(+:result)
    for (int i = 0; i < n; i++) {
        result += a[i],n;
    }
    
    res[0] = result;
    
}

