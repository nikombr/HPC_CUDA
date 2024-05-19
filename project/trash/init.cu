
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>

                                                                                                                                                                                                                                                                                                                                   

void init(double *** u, double *** uold, double *** f, int N, double start_T, int kstart, int kend) {

    int kwidth = kend - kstart + 1; // Equal to N in case of no split


    if (kwidth == N) {

        // Initialize values
        double delta = 2.0/(N+1);
        double fracdelta = (N+1)/2.0;

        // Set f to zero everywhere
        memset(f[0][0],0,(N+2)*(N+2)*(N+2)*sizeof(double));

        // Overwrite a specific region
        int ux = floor(0.625*fracdelta), uy = floor(0.5*fracdelta), lz = ceil(1.0/3.0*fracdelta), uz = floor(fracdelta);
        for (int i = 1; i <= ux; i++) {
            for (int j = 1; j <= uy; j++) {
                for (int k = lz; k <= uz; k++) {   
                    f[i][j][k] = 200;
                }
            }
        }

        // Initialize uold to start_T
        for (int i = 1; i < N+1; i++) {
            for (int j = 1; j < N+1; j++) {
                for (int k = 1; k < N+1; k++) {
                    uold[i][j][k] = start_T;
                    u[i][j][k] = start_T;
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
    else {

        // Initialize values
        double delta = 2.0/(N+1);
        double fracdelta = (N+1)/2.0;

        // Set f to zero everywhere
        memset(f[0][0], 0, (N + 2) * (N + 2) * (kwidth + 2) * sizeof(double));

        // Overwrite a specific region
        int ux = floor(0.625*fracdelta), uy = floor(0.5*fracdelta), lz = ceil(1.0/3.0*fracdelta), uz = floor(fracdelta);
        if (kstart <= uz && kend > lz) {
            for (int i = 1; i <= ux; i++) {
                for (int j = 1; j <= uy; j++) {
                    for (int k = max(lz - kstart + 1, 0); k <= min(uz - kstart + 1, kwidth + 1); k++) {   
                        f[i][j][k] = 200;
                    }
                }
            }
        }

        // Initialize uold to start_T
        for (int i = 1; i < N+1; i++) {
            for (int j = 1; j < N+1; j++) {
                for (int k = 0; k < kwidth + 2; k++) { // We just initialize everything and overwrite the boundary after
                    uold[i][j][k] = start_T;
                    u[i][j][k] = start_T;
                }
            }
        }

        // Set the boundary of uold and u
        if (kend == N) {
            for (int i = 0; i < N+2; i++) {
                for (int j = 0; j < N+2; j++) {

                    uold[i][j][kwidth + 1] = 20.0;
                    u[i][j][kwidth + 1] = 20.0;

                }
            }
        }
        else if (kstart == 1) {
            for (int i = 0; i < N+2; i++) {
                for (int j = 0; j < N+2; j++) {

                    uold[i][j][0] = 20.0;
                    u[i][j][0] = 20.0;

                }
            }
        }

        for (int j = 0; j < N+2; j++) {
            for (int k = 0; k < kwidth + 2; k++) {

                uold[0][j][k] = 20.0;
                uold[N+1][j][k] = 20.0;

                u[0][j][k] = 20.0;
                u[N+1][j][k] = 20.0;
            }
        }

        for (int i = 0; i < N+2; i++) {
            for (int k = 0; k < kwidth + 2; k++) {

                uold[i][0][k] = 0;
                uold[i][N+1][k] = 20.0;

                u[i][0][k] = 0;
                u[i][N+1][k] = 20.0;
            }
        }

    }

}