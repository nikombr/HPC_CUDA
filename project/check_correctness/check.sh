#!/bin/bash
#BSUB -J check # name
#BSUB -o outfiles/close_%J.out # output file
#BSUB -q gpuh100
#BSUB -n 32 ## cores
#BSUB -R "rusage[mem=1GB]" 
#BSUB -W 60 # useable time in minutes
##BSUB -N # send mail when done
#BSUB -R "span[hosts=1]"

## Script that tests if the implementations yield the correct result

module load python3/3.10.13


ITER=2000
N=50
TOLERANCE=-1
START_T=5

CPUFOLDER="../cpu/"
GPUFOLDER="../gpu/"

threads=32

echo ""
echo "CPU, reduction"
OMP_NUM_THREADS=$threads OMP_SCHEDULE=static OMP_PROC_BIND=close OMP_PLACES=cores ./../cpu/jacobi_reduction $N $ITER $TOLERANCE $START_T 3 1
echo ""
echo "CPU, no reduction"
OMP_NUM_THREADS=$threads OMP_SCHEDULE=static OMP_PROC_BIND=close OMP_PLACES=cores ./../cpu/jacobi_no_reduction $N $ITER $TOLERANCE $START_T 3 2

echo ""
echo "GPU, reduction"
./../gpu/jacobi_reduction $N $ITER $TOLERANCE $START_T 3 1 # Reduction, best
echo ""
echo "GPU, no reduction"
./../gpu/jacobi_no_reduction $N $ITER $TOLERANCE $START_T 3 2 # No reduction
echo ""
echo "GPU, reduction (atomic)"
./../gpu/jacobi_reduction_atomic $N $ITER $TOLERANCE $START_T 3 3 # Reduction, atomic add

echo ""
echo ""
echo "Error (they should all be zero)"
echo ""

python3 ./binary_cmp.py results/poisson_cpu_${N}_no_reduction.bin results/poisson_cpu_${N}_reduction.bin 
python3 ./binary_cmp.py results/poisson_cpu_${N}_no_reduction.bin results/poisson_gpu_${N}_no_reduction.bin 
python3 ./binary_cmp.py results/poisson_gpu_${N}_reduction.bin results/poisson_cpu_${N}_reduction.bin 
python3 ./binary_cmp.py results/poisson_gpu_${N}_reduction_atomic.bin results/poisson_cpu_${N}_reduction.bin 

exit 0

