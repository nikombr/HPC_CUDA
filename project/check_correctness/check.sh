#!/bin/bash

module load python3/3.10.13


ITER=1000
N=50
TOLERANCE=-1
START_T=5

CPUFOLDER="../cpu/"
GPUFOLDER="../gpu/"

OMP_NUM_THREADS=$threads OMP_SCHEDULE=static OMP_PROC_BIND=close OMP_PLACES=cores ./$CPUFOLDERjacobi_reduction $N $ITER $TOLERANCE $START_T 3 reduction
OMP_NUM_THREADS=$threads OMP_SCHEDULE=static OMP_PROC_BIND=close OMP_PLACES=cores ./$CPUFOLDERjacobi_no_reduction $N $ITER $TOLERANCE $START_T 3 no_reduction
    
CUDA_VISIBLE_DEVICES=0 ./$GPUFOLDERjacobi_reduction $N $ITER $TOLERANCE $START_T 3 reduction
CUDA_VISIBLE_DEVICES=0 ./$GPUFOLDERjacobi_no_reduction $N $ITER $TOLERANCE $START_T 3 no_reduction

##python3 ./binary_cmp.py 

exit 0

