#ifndef _INFO_STRUCT_H
#define _INFO_STRUCT_H

struct Info { // Stores info about MPI and device structure
    double ***u_d;
    double ***uold_d;
    double ***f_d;
    double *u_log;
    double *uold_log;
    double *f_log;
    int N;
    int iter_max;
    double *tolerance;
    int *n; 
    int *iter;
    int world_size;
    int world_rank;
    int local_rank;
    int canAccessPeerPrev;
    int canAccessPeerNext;
    int istart;
    int iend;
    int iwidth;
};

#endif