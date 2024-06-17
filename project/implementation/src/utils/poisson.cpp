#include "../../lib/poisson.h"


Poisson::Poisson(int N, int GPU, double start_T, int iter_max, double tolerance) {
    this->N                 = N;
    this->GPU               = GPU;
    this->deviceData[0].width = N;
    this->deviceData[0].start = 1;
    this->deviceData[0].end   = N;
    this->deviceData[0].peer_width = 0;
    this->deviceData[0].canAccesPeerNext = false;
    this->deviceData[0].canAccesPeerPrev = false;
    this->deviceData[1].canAccesPeerNext = false;
    this->deviceData[1].canAccesPeerPrev = false;
    this->start_T           = start_T;
    this->iter_max          = iter_max;
    this->tolerance         = tolerance;
}