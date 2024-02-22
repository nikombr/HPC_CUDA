
#include <stdio.h>
#include <stdlib.h>

__global__ void reduction_baseline(double *a, int n, double *res) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) atomicAdd(res, a[idx]);
}
