#ifndef _POISSON_H
#define _POISSON_H

class Poisson {
    private:        
        int canAccessPeerPrev = false;
        int canAccessPeerNext = false;
        int start; 
        int end;
        int GPU;
        int num_device;
        double start_T;
    public:
        int width;
        int world_size;
        int world_rank;
        int local_rank;
        int N;
        double tolerance;
        int n; 
        int iter;
        int iter_max;
        double ***u_d;
        double ***uold_d;
        double ***f_d;
        double ***u_h;
        double ***uold_h;
        double ***f_h;
        double *u_log;
        double *uold_log;
        double *f_log;
        char *output_prefix;
        double time_nccl_setup = 0;
        double time_nccl_transfer = 0;
        double P2P = 0; // -1 = P2P to previous, 0 no P2P, 1 P2P to next
        Poisson(int N, int GPU, double start_T, int iter_max, double tolerance);// Sets default values 
        void alloc();                                                           // Allocates matrices
        void init();                                                            // Initializes matrices
        void jacobi();                                                          // Computes Jacobi-iterations
        void swapArrays();                                                      // Swaps arrays in jacobi
        void finalize(int output_type, char*output_ext, char *extra_str);       // Frees matrices, dumps output
        void sendToDevice();                                                    // Sends matrices to device
        void sendToHost();                                                      // Sends matrix to host
        void setupMultipleGPU(int print);                                       // Sets up multiple GPUs and MPI
        void setup_f_matrix();
};

#endif