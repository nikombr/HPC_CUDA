#ifndef __MAPS_H
#define __MAPS_H


__host__ __device__
double map_pi(double x, int n) {
    x = (x + 0.5) / n;
    return 4.0 / (n * (1.0 + x * x));
}
__host__ __device__
    double map_softmax(double x, int n) {
    return exp((x + 1.0) / n);
}

#define map map_pi


#endif
