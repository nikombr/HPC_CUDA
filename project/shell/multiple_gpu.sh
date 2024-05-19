#!/bin/bash
#BSUB -J mgpu # name
#BSUB -o outfiles/mgpu_%J.out # output file
#BSUB -q gpuh100
#BSUB -n 16 ## cores
#BSUB -R "rusage[mem=1GB]" 
#BSUB -W 5 # useable time in minutes
##BSUB -N # send mail when done
#BSUB -R "span[ptile=16]"
#BSUB -gpu "num=2:mode=exclusive_process"

ITER=2000
TOLERANCE=-1
START_T=5

## lscpu


ARCH=`uname -m`

module load gcc

module load cuda/12.2.2

module load mpi/5.0.2-gcc-12.3.0-binutils-2.40

module load nccl/2.19.3-1-cuda-12.2.2 

## make clean
## make realclean
## make

if [[ "$ARCH" == "aarch64" ]]
then
    GPU="gracy"
    THREADS=$(seq 1 1 72)
    echo "Running on gracy :)"
else
    if [[ "$ARCH" == "x86_64" ]]
    then
        GPU="gpuh100"
        THREADS=$(seq 1 1 32)
        echo "Running on gpuh100"
    else
        echo "Confused!"
        exit 1
    fi
fi

FOLDER="../results/mgpu/${GPU}"
EXECUTEFOLDER="../implementation/execute/"

FILE_REDUCTION=$FOLDER/reduction.txt
FILE_NO_REDUCTION=$FOLDER/no_reduction.txt
FILE_REDUCTION_ATOMIC=$FOLDER/reduction_atomic.txt

rm -rf $FILE_REDUCTION
rm -rf $FILE_NO_REDUCTION
rm -rf $FILE_REDUCTION_ATOMIC

N=50

mpirun -npernode 2 ./${EXECUTEFOLDER}mgpu_no_reduction $N $ITER $TOLERANCE $START_T >> $FILE_NO_REDUCTION

mpirun -npernode 1 ./${EXECUTEFOLDER}mgpu_no_reduction $N $ITER $TOLERANCE $START_T >> $FILE_NO_REDUCTION

## for N in {10..500..10} {520..600..20} {630..690..30};
## do

    ## mpirun -npernode 1 ./${EXECUTEFOLDER}mgpu_no_reduction $N $ITER $TOLERANCE $START_T >> $FILE_NO_REDUCTION
    ## ./${EXECUTEFOLDER}mgpu_reduction_atomic $N $ITER $TOLERANCE $START_T >> $FILE_REDUCTION_ATOMIC
    ##./${EXECUTEFOLDER}mgpu_reduction $N $ITER $TOLERANCE $START_T >> $FILE_REDUCTION

## done
exit 0

