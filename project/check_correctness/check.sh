#!/bin/bash
#BSUB -J check # name
#BSUB -o outfiles/check_%J.out # output file
#BSUB -q gpuh100
#BSUB -n 16 ## cores
#BSUB -R "rusage[mem=1GB]" 
#BSUB -W 30 # useable time in minutes
##BSUB -N # send mail when done
#BSUB -R "span[ptile=8]"
#BSUB -gpu "num=2:mode=exclusive_process"

## Script that tests if the implementations yield the correct result

module load python3/3.10.13
module load gcc
module load cuda/12.2.2
module load mpi/5.0.2-gcc-12.3.0-binutils-2.40

module load nccl/2.19.3-1-cuda-12.2.2

pip3 install matplotlib


ITER=1000
N=100
TOLERANCE=-1
START_T=5

CPUFOLDER="../cpu/"
GPUFOLDER="../gpu/"

threads=32

echo ""
echo "CPU, reduction"
OMP_NUM_THREADS=$threads OMP_SCHEDULE=static OMP_PROC_BIND=close OMP_PLACES=cores ./../implementation/execute/cpu_reduction $N $ITER $TOLERANCE $START_T 3 1
echo ""
echo "CPU, no reduction"
OMP_NUM_THREADS=$threads OMP_SCHEDULE=static OMP_PROC_BIND=close OMP_PLACES=cores ./../implementation/execute/cpu_no_reduction $N $ITER $TOLERANCE $START_T 3 2

echo ""
echo "GPU, reduction"
./../implementation/execute/gpu_reduction $N $ITER $TOLERANCE $START_T 3 1 # Reduction, best
echo ""
echo "GPU, no reduction"
./../implementation/execute/gpu_no_reduction $N $ITER $TOLERANCE $START_T 3 2 # No reduction
echo ""
echo "GPU, reduction (atomic)"
./../implementation/execute/gpu_reduction_atomic $N $ITER $TOLERANCE $START_T 3 3 # Reduction, atomic 

echo ""
echo "MGPU, no reduction"
## NCCL_DEBUG=INFO mpirun -npernode 2 ./../implementation/execute/mgpu_no_reduction $N $ITER $TOLERANCE $START_T 3 2 # Reduction, atomic add
mpirun -npernode 2 ./../implementation/execute/mgpu_no_reduction $N $ITER $TOLERANCE $START_T 3 2 # Reduction, atomic add

echo ""
echo ""
echo "Error (they should all be zero)"
echo ""

python3 ./binary_cmp.py results/poisson_cpu_${N}_no_reduction results/poisson_cpu_${N}_reduction 1
python3 ./binary_cmp.py results/poisson_cpu_${N}_no_reduction results/poisson_gpu_${N}_no_reduction 2
python3 ./binary_cmp.py results/poisson_gpu_${N}_reduction results/poisson_cpu_${N}_reduction 3
python3 ./binary_cmp.py results/poisson_gpu_${N}_reduction_atomic results/poisson_cpu_${N}_reduction 4
python3 ./binary_cmp.py results/poisson_mgpu_${N}_no_reduction results/poisson_cpu_${N}_reduction 5

exit 0

