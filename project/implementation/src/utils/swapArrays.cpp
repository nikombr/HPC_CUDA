#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../../lib/poisson.h"


void Poisson::swapArrays() {

    double ***tmp;

    if (GPU) {
        double *tmp_log;
        for (int i = 0; i < num_device_per_process; i++) {
            tmp = deviceData[i].u_d;
            deviceData[i].u_d = deviceData[i].uold_d;
            deviceData[i].uold_d = tmp;
            tmp_log = deviceData[i].u_log;
            deviceData[i].u_log = deviceData[i].uold_log;
            deviceData[i].uold_log = tmp_log;
        }
    }
    else {
        tmp = u_h;
        u_h = uold_h;
        uold_h = tmp;

    }
}