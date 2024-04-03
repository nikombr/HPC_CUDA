#ifndef __ALLOC_3D
#define __ALLOC_3D

double *** host_malloc_3d(int m, int n, int k);
void device_malloc_3d(double ****B,double **b,int m, int n, int k);

#define HAS_FREE_3D
void host_free_3d(double ***p);
void device_free_3d(double ***B, double *b);

#endif /* __ALLOC_3D */
