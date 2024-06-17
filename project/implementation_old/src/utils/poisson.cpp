#include "../../lib/poisson.h"

Poisson::Poisson(int N, int GPU, double start_T, int iter_max, double tolerance) {
    this->N = N;
    this->GPU = GPU;
    this->width = N;
    this->start = 1;
    this->end = N;
    this->start_T = start_T;
    this->iter_max = iter_max;
    this->tolerance = tolerance;
    this->iter = 0;
    this->n = 0;
    this->world_rank = 0; // If we are only runnning on one it defaults
    this->local_rank = 0; // If we are only runnning on one it defaults
}