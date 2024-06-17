#include "../../lib/poisson.h"
#include "../../lib/malloc3d.h"
#include "../../lib/cudaMalloc3d.h"

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#ifdef _NORMAL_MAKE
    #include <mpi.h>
#endif
#include <cuda_runtime_api.h>


void Poisson::alloc() {
    if (GPU) { // GPU
        // Allocation on host
        if (world_rank == 0) { // Allocate more space, so we can do transfer to rank 0 with all data
            uold_h  = host_malloc_3d(N + 2, N + 2, N + 2);
        }
        else {
            uold_h  = host_malloc_3d(total_width + 2, N + 2, N + 2);
        }
        
        // Check allocation on host
        if (uold_h == NULL) {
            perror("Allocation of uold failed on host!");
            exit(-1);
        }

        // Allocation on device
        for (int i = 0; i < num_device_per_process; i++) {
            cudaSetDevice(i);
            device_malloc_3d(&deviceData[i].u_d,    &deviceData[i].u_log,    deviceData[i].width + 2 - (num_device_per_process > 1), N + 2, N + 2);
            device_malloc_3d(&deviceData[i].uold_d, &deviceData[i].uold_log, deviceData[i].width + 2 - (num_device_per_process > 1), N + 2, N + 2);
            // Check allocation on device
            if (deviceData[i].u_d == NULL || deviceData[i].uold_d == NULL || deviceData[i].u_log == NULL || deviceData[i].uold_log == NULL) {
                fprintf(stderr,"Allocation of u and/or uold failed on device %d!",i);
                exit(-1);
            }
        }
        
    }
    else { // CPU
        // Allocation on host
        u_h     = malloc_3d(N + 2, N + 2, N + 2);
        uold_h  = malloc_3d(N + 2, N + 2, N + 2);
        
        // Check allocation on host
        if (u_h == NULL || uold_h == NULL) {
            perror("Allocation of u and/or uold failed on host!");
            exit(-1);
        }
    }
    return;
}