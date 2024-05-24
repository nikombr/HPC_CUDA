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
        double time_iterations = 0;
        double time_loop = 0;
        Poisson(int N, int GPU, double start_T, int iter_max, double tolerance);
        void setupMultipleGPU(int print);
        void alloc();
        void init();
        void sendToDevice();
        void jacobi();
        void sendToHost();
        void swapArrays();
        void finalize(int output_type, char*output_ext, char *extra_str);
};

#endif