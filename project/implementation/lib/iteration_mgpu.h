#ifndef _ITERATION_H
#define _ITERATION_H
#include <cuda_runtime_api.h>
void iteration(double *** u, double *** uold, double *** uold_peer, double *** f, int N, int width, int peer_width, int canAccessPeerPrev, int canAccessPeerNext, cudaStream_t  stream);

#endif