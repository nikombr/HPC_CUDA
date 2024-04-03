#include <stdlib.h>

double ***
host_malloc_3d(int m, int n, int k) {

    double ***p;
    double *a;

    if (m <= 0 || n <= 0 || k <= 0)
        return NULL;

    cudaMallocHost(&p, m * sizeof(double **) + m * n * sizeof(double *));

    if (p == NULL) {
        return NULL;
    }

    for(int i = 0; i < m; i++) {
        p[i] = (double **) p + m + i * n ;
    }

    cudaMallocHost(&a, m * n * k * sizeof(double));

    if (a == NULL) {
        cudaFreeHost(p);
        return NULL;
    }

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            p[i][j] = a + (i * n * k) + (j * k);
        }
    }

    return p;
}

void
host_free_3d(double ***p) {
    cudaFreeHost(p[0][0]);
    cudaFreeHost(p);
}

__global__ void mallocLoops(double***p, double *a,int m, int n, int k) {
    for(int i = 0; i < m; i++) {
        p[i] = (double **) p + m + i * n;
    }

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            p[i][j] = a + (i * n * k) + (j * k);
        }
    }
}

void
device_malloc_3d(double ****B,double ** b,int m, int n, int k) {

    double ***p;
    double *a;

    if (m <= 0 || n <= 0 || k <= 0)
        *B = NULL;

    cudaMalloc(&p, m * sizeof(double **) + m * n * sizeof(double *));

    if (p == NULL) {
        *B = NULL;
    }

    cudaMalloc(&a, m * n * k * sizeof(double));

    if (a == NULL) {
        cudaFree(p);
        *B = NULL;
    }

    mallocLoops<<<1,1>>>(p, a, m, n, k);

    *B = p;
    *b = a;
}

void device_free_3d(double ***B,double*b) {
    cudaFree(b);
    cudaFree(B);
}