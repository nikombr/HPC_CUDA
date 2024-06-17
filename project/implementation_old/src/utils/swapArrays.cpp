#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../../lib/poisson.h"


void Poisson::swapArrays() {

    double ***tmp;

    if (this->GPU) {
        double *tmp_log;
        tmp = this->u_d;
        this->u_d = this->uold_d;
        this->uold_d = tmp;
        tmp_log = this->u_log;
        this->u_log = this->uold_log;
        this->uold_log = tmp_log;
    }
    else {
        tmp = this->u_h;
        this->u_h = this->uold_h;
        this->uold_h = tmp;

    }
}