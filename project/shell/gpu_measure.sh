#!/bin/bash
#BSUB -J gpu # name
#BSUB -o outfiles/gpu_measure_%J.out # output file
#BSUB -q gpuh100
#BSUB -n 32 ## cores
#BSUB -R "rusage[mem=1GB]" 
#BSUB -W 24:00 # useable time in minutes
##BSUB -N # send mail when done
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"

ITER=1000
TOLERANCE=-1
START_T=5

## lscpu

ARCH=`uname -m`



## make clean
## make realclean
## make

if [[ "$ARCH" == "aarch64" ]]
then
    GPU="gracy"
    echo "Running on gracy :)"
else
    if [[ "$ARCH" == "x86_64" ]]
    then
        module load gcc

        module load cuda/12.2.2

        module load mpi/5.0.2-gcc-12.3.0-binutils-2.40

        module load nccl/2.19.3-1-cuda-12.2.2 
        
        GPU="gpuh100"
        echo "Running on gpuh100"
    else
        echo "Confused!"
        exit 1
    fi
fi

FOLDER="../results/gpu/${GPU}"
EXECUTEFOLDER="../implementation/execute/"

FILE_REDUCTION=$FOLDER/reduction.txt
FILE_NO_REDUCTION=$FOLDER/no_reduction.txt
FILE_REDUCTION_ATOMIC=$FOLDER/reduction_atomic.txt

rm -rf $FILE_REDUCTION
rm -rf $FILE_NO_REDUCTION
rm -rf $FILE_REDUCTION_ATOMIC

## for N in {10..500..10} {520..600..20} {630..690..30};
for N in {10..90..10} {100..200..25} {250..1400..50};
do

    ./${EXECUTEFOLDER}gpu_reduction $N $ITER $TOLERANCE $START_T >> $FILE_REDUCTION
    ./${EXECUTEFOLDER}gpu_reduction_atomic $N $ITER $TOLERANCE $START_T >> $FILE_REDUCTION_ATOMIC
    ./${EXECUTEFOLDER}gpu_no_reduction $N $ITER $TOLERANCE $START_T >> $FILE_NO_REDUCTION

done
exit 0

