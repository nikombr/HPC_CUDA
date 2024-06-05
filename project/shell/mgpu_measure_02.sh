#!/bin/bash
#BSUB -J mgpu02 # name
#BSUB -o outfiles/mgpu02_%J.out # output file
#BSUB -q gpuh100
#BSUB -n 32 ## cores
#BSUB -R "rusage[mem=1GB]" 
#BSUB -W 500 # useable time in minutes
##BSUB -N # send mail when done
#BSUB -R "span[ptile=32]"
#BSUB -gpu "num=2:mode=exclusive_process"

ITER=1000
TOLERANCE=-1
START_T=5

## lscpu

ARCH=`uname -m`

module load gcc/12.3.0-binutils-2.40
module load cuda/12.2.2
module load mpi/5.0.3-gcc-12.3.0-binutils-2.40
module load nccl/2.19.3-1-cuda-12.2.2 

## make clean
## make realclean
## make

echo "Running on gpuh100"

FOLDER="../results/mgpu"
EXECUTEFOLDER="../implementation/execute/"

FILE_NO_REDUCTION=$FOLDER/no_reduction/result02.txt
FILE_REDUCTION=$FOLDER/reduction/result02.txt

rm -rf $FILE_NO_REDUCTION
rm -rf $FILE_REDUCTION

##for N in {10..500..10} {520..600..20} {630..690..30};{10..90..10} {100..200..25} {250..900..50} 
for N in {10..90..10} {100..200..25} {250..1250..50};
do
    mpirun -npernode 2 ./${EXECUTEFOLDER}mgpu_no_reduction $N $ITER $TOLERANCE $START_T >> $FILE_NO_REDUCTION
    mpirun -npernode 2 ./${EXECUTEFOLDER}mgpu_reduction $N $ITER $TOLERANCE $START_T >> $FILE_REDUCTION
done
exit 0

