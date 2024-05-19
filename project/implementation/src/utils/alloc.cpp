#include "../../lib/poisson.h"
#include "../../lib/malloc3d.h"
#include "../../lib/cudaMalloc3d.h"

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#include <cuda_runtime_api.h>


void Poisson::alloc() {
    if (this->GPU) { // GPU
        // Allocation on host
        if (world_rank == 0) { // Allocate more space, so we can do transfer to rank 0 with all data
            this->u_h  = host_malloc_3d(this->N+2, this->N+2, this->N+2);
            this->uold_h  = host_malloc_3d(this->N+2, this->N+2, this->N+2);
            this->f_h     = host_malloc_3d(this->N+2, this->N+2, this->N+2);
        }
        else {
            this->u_h     = host_malloc_3d(this->width+2, this->N+2, this->N+2);
            this->uold_h  = host_malloc_3d(this->width+2, this->N+2, this->N+2);
            this->f_h     = host_malloc_3d(this->width+2, this->N+2, this->N+2);
        }
        
        
        // Check allocation on host
        if (this->u_h == NULL || this->uold_h == NULL || this->f_h == NULL) {
            perror("Allocation failed on host!");
            exit(-1);
        }

        // Allocation on device
        device_malloc_3d(&this->u_d,    &this->u_log,    this->width+2, this->N+2, this->N+2);
        device_malloc_3d(&this->uold_d, &this->uold_log, this->width+2, this->N+2, this->N+2);
        device_malloc_3d(&this->f_d,    &this->f_log,    this->width+2, this->N+2, this->N+2);

        // Check allocation on device
        if (this->u_log == NULL || this->uold_log == NULL || this->f_log == NULL || this->u_d == NULL || this->uold_d == NULL || this->f_d == NULL) {
            perror("Allocation failed on device!");
            exit(-1);
        }
    }
    else { // CPU
        // Allocation on host
        this->u_h     = malloc_3d(this->N+2, this->N+2, this->N+2);
        this->uold_h  = malloc_3d(this->N+2, this->N+2, this->N+2);
        this->f_h     = malloc_3d(this->N+2, this->N+2, this->N+2);
        
        // Check allocation on host
        if (this->u_h == NULL || this->uold_h == NULL || this->f_h == NULL) {
            perror("Allocation failed on host!");
            exit(-1);
        }
    }
    return;
}