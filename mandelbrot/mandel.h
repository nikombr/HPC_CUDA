#ifndef __MANDEL_H
#define __MANDEL_H

__global__ void mandel(int width, int height, int *image, int max_iter);
__global__ void mandel_2d(int width, int height, int *image, int max_iter);
void mandel_omp(int disp_width, int disp_height, int *array, int max_iter);

#endif
