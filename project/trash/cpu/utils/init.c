
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>

                                                                                                                                                                                                                                                                                                                                   

void omp_get_schedule(omp_sched_t *kind, int *chunk_size)
{
*kind = omp_sched_static;
*chunk_size = 0;
}

void init(double *** u, double *** uold, double *** f, int N, double start_T) {

    // Initialize values
    double delta = 2.0/(N+1);
    double fracdelta = (N+1)/2.0;

    if (omp_get_max_threads() > 1) {
        #pragma omp parallel for schedule(runtime)
        for (int i = 0; i <= N+1; i++) {
            for (int j = 0; j <= N+1; j++) {
                for (int k = 0; k <= N+1; k++) {  
                    // Set f to zero everywhere 
                    f[i][j][k] = 0;
                    // Initialize uold to start_T
                    uold[i][j][k] = start_T;
                    u[i][j][k] = start_T;
                }
            }
        }
    }
    else {
        // Set f to zero everywhere
        memset(f[0][0],0,(N+2)*(N+2)*(N+2)*sizeof(double));
        // Initialize uold to start_T
        for (int i = 1; i < N+1; i++) {
            for (int j = 1; j < N+1; j++) {
                for (int k = 1; k < N+1; k++) {
                    uold[i][j][k] = start_T;
                    u[i][j][k] = start_T;
                }
            }
        }
    }

    // Overwrite a specific region
    int ux = floor(0.625*fracdelta), uy = floor(0.5*fracdelta), lz = ceil(1.0/3.0*fracdelta), uz = floor(fracdelta);
    for (int i = 1; i <= ux; i++) {
        for (int j = 1; j <= uy; j++) {
            for (int k = lz; k <= uz; k++) {   
                f[i][j][k] = 200;
            }
        }
    }

    // Set the boundary of uold and u
    for (int i = 0; i < N+2; i++) {
        for (int j = 0; j < N+2; j++) {

            uold[0][j][i] = 20.0;
            uold[N+1][j][i] = 20.0;

            u[0][j][i] = 20.0;
            u[N+1][j][i] = 20.0;

            uold[i][0][j] = 0;
            uold[i][N+1][j] = 20.0;

            u[i][0][j] = 0;
            u[i][N+1][j] = 20.0;

            uold[i][j][0] = 20.0;
            uold[i][j][N+1] = 20.0;

            u[i][j][0] = 20.0;
            u[i][j][N+1] = 20.0;
        }
    }

}