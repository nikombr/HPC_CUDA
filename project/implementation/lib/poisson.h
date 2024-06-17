#ifndef _POISSON_H
#define _POISSON_H

struct DeviceData {
    double ***u_d;
    double ***uold_d;
    double ***uold_peer;
    double ***f_d;
    double *u_log;
    double *uold_log;
    double *f_log;
    int width;
    int start;
    int end;
    int rank;
    int first;
    int last;
    int peer_width;
    int canAccesPeerPrev;
    int canAccesPeerNext;
    // Constants for nccl transfers
    int firstRowReceive;
    int lastRowSend;
    int firstRowSend;
    int lastRowReceive;
};

class Poisson {
    private:        
        int GPU;
        int num_device = 1;
        int num_device_per_process = 1;
        double start_T;
        struct DeviceData deviceData[2]; // Maximum of 2 devices per process in this setup
        int iter_max;
        double ***u_h;
        double ***uold_h;
        double ***f_h;
        int total_width = 0;
    public:
        int world_size = 1;
        int world_rank = 0;
        int N;
        double tolerance;
        int n = 0; 
        int iter = 0;
        char *output_prefix;
        double time_nccl_setup = 0;
        double time_nccl_transfer = 0;
        Poisson(int N, int GPU, double start_T, int iter_max, double tolerance);// Sets default values 
        void alloc();                                                           // Allocates matrices
        void setup_f_matrix();                                                  // Allocates matrices
        void init();                                                            // Initializes matrices
        void jacobi();                                                          // Computes Jacobi-iterations
        void swapArrays();                                                      // Swaps arrays in jacobi
        void finalize(int output_type, char*output_ext, char *extra_str);       // Frees matrices, dumps output
        void sendToDevice();                                                    // Sends matrices to device
        void sendToHost();                                                      // Sends matrix to host
        void setupMultipleGPU(int print);                                       // Sets up multiple GPUs and MPI
};

#endif