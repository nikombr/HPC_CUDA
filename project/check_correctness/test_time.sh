#!/bin/bash
#BSUB -J check # name
#BSUB -o outfiles/check_%J.out # output file
#BSUB -q gpuh100
#BSUB -n 32 ## cores
#BSUB -R "rusage[mem=1GB]" 
#BSUB -W 24:00 # useable time in minutes
##BSUB -N # send mail when done
#BSUB -R "span[ptile=32]"
#BSUB -gpu "num=2:mode=exclusive_process"

## Script that tests if the implementations yield the correct result

module load python3/3.10.13
module load gcc
module load cuda/12.2.2
module load mpi/5.0.2-gcc-12.3.0-binutils-2.40

module load nccl/2.19.3-1-cuda-12.2.2

pip3 install matplotlib


ITER=2
N=800
TOLERANCE=-1
START_T=5

CPUFOLDER="../cpu/"
GPUFOLDER="../gpu/"

threads=32

## echo ""
## echo "CPU, no reduction"
## OMP_NUM_THREADS=$threads OMP_SCHEDULE=static OMP_PROC_BIND=close OMP_PLACES=cores ./../implementation/execute/cpu_no_reduction $N $ITER $TOLERANCE $START_T 

## echo ""
## echo "CPU, reduction"
## OMP_NUM_THREADS=$threads OMP_SCHEDULE=static OMP_PROC_BIND=close OMP_PLACES=cores ./../implementation/execute/cpu_reduction $N $ITER $TOLERANCE $START_T

echo ""
echo "GPU, no reduction"
mpirun -npernode 1 ./../implementation/execute/gpu_no_reduction $N $ITER $TOLERANCE $START_T # No reduction
echo ""
echo "GPU, reduction"
./../implementation/execute/gpu_reduction $N $ITER $TOLERANCE $START_T # Reduction, best
## echo ""
## echo "GPU, reduction (atomic)"
## ./../implementation/execute/gpu_reduction_atomic $N $ITER $TOLERANCE $START_T # Reduction, atomic 

echo ""
echo "MGPU, no reduction"
## NCCL_DEBUG=INFO mpirun -npernode 2 ./../implementation/execute/mgpu_no_reduction $N $ITER $TOLERANCE $START_T 3 2
NCCL_DEBUG=INFO mpirun -npernode 1 ./../implementation/execute/mgpu_no_reduction $N $ITER $TOLERANCE $START_T  # No reduction

echo ""
echo "MGPU, no reduction, asynchronus"
mpirun -npernode 1 ./../implementation/execute/mgpu_reduction $N $ITER $TOLERANCE $START_T # Reduction

exit 0

