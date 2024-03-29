
#include <stdio.h>
#include <stdlib.h>

__global__ void mandel(int disp_width, int disp_height, int *array, int max_iter) {

    double 	scale_real, scale_imag;
    double 	x, y, u, v, u2, v2;
    int 	i, j, iter;

    scale_real = 3.5 / (double)disp_width;
    scale_imag = 3.5 / (double)disp_height;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	x = ((double)i * scale_real) - 2.25; 
	if (i < disp_width) {
		for (j = 0; j < disp_height; j++) {
			y = ((double)j * scale_imag) - 1.75; 

			u    = 0.0;
			v    = 0.0;
			u2   = 0.0;
			v2   = 0.0;
			iter = 0;

			while ( u2 + v2 < 4.0 &&  iter < max_iter ) {
				v = 2 * v * u + y;
				u = u2 - v2 + x;
				u2 = u*u;
				v2 = v*v;
				iter = iter + 1;
			}

			// if we exceed max_iter, reset to zero
			iter = iter == max_iter ? 0 : iter;

			array[i*disp_height + j] = iter;
		}
	}

}

__global__ void mandel_2d(int disp_width, int disp_height, int *array, int max_iter) {

    double 	scale_real, scale_imag;
    double 	x, y, u, v, u2, v2;
    int 	i, j, iter;

    scale_real = 3.5 / (double)disp_width;
    scale_imag = 3.5 / (double)disp_height;

	i = threadIdx.x + blockIdx.x * blockDim.x;
	j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i < disp_width && j < disp_height) {

		x = ((double)i * scale_real) - 2.25; 
		y = ((double)j * scale_imag) - 1.75; 

		u    = 0.0;
		v    = 0.0;
		u2   = 0.0;
		v2   = 0.0;
		iter = 0;

		while ( u2 + v2 < 4.0 &&  iter < max_iter ) {
			v = 2 * v * u + y;
			u = u2 - v2 + x;
			u2 = u*u;
			v2 = v*v;
			iter = iter + 1;
		}

		// if we exceed max_iter, reset to zero
		iter = iter == max_iter ? 0 : iter;

		array[i*disp_height + j] = iter;
	}

}

void mandel_omp(int disp_width, int disp_height, int *array, int max_iter) {

    double 	scale_real, scale_imag;
    double 	x, y, u, v, u2, v2;
    int 	i, j, iter;

    scale_real = 3.5 / (double)disp_width;
    scale_imag = 3.5 / (double)disp_height;

	#pragma omp parallel for schedule(dynamic) private(x, y, u, v, u2, v2, i, j, iter)
    for (i = 0; i < disp_width; i++) {
		
	x = ((double)i * scale_real) - 2.25; 

		for (j = 0; j < disp_height; j++) {
			y = ((double)j * scale_imag) - 1.75; 

			u    = 0.0;
			v    = 0.0;
			u2   = 0.0;
			v2   = 0.0;
			iter = 0;

			while ( u2 + v2 < 4.0 &&  iter < max_iter ) {
				v = 2 * v * u + y;
				u = u2 - v2 + x;
				u2 = u*u;
				v2 = v*v;
				iter = iter + 1;
			}

			// if we exceed max_iter, reset to zero
			iter = iter == max_iter ? 0 : iter;

			array[i*disp_height + j] = iter;
		}
    }


}
