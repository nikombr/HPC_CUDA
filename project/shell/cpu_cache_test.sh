#!/bin/bash
#BSUB -J cpu_cache_test # name
#BSUB -o outfiles/cpu_cache_test_%J.out # output file
#BSUB -q gpuh100
#BSUB -n 32 ## cores
#BSUB -R "rusage[mem=1GB]" 
#BSUB -W 500 # useable time in minutes
##BSUB -N # send mail when done
#BSUB -R "span[hosts=1]"

ITER=1000
TOLERANCE=-1
START_T=5

ARCH=`uname -m`

## make clean
## make realclean
## make


if [[ "$ARCH" == "aarch64" ]]
then
    CPU="gracy"
    THREADS="32 72"
    echo "Running on gracy :)"
else
    if [[ "$ARCH" == "x86_64" ]]
    then
        module load gcc

        module load cuda/12.2.2

        module load mpi/5.0.2-gcc-12.3.0-binutils-2.40

        module load nccl/2.19.3-1-cuda-12.2.2 
        CPU="gpuh100"
        THREADS="16 32"
        echo "Running on gpuh100"
    else
        echo "Confused!"
        exit 1
    fi
fi

FOLDER="../results/cpu/$CPU/cache_test"
FOLDER2="../results/cpu_spread/$CPU/cache_test"
EXECUTEFOLDER="../implementation/execute/"

FILE_SPREAD_REDUCTION=$FOLDER2/reduction.txt
FILE_SPREAD_NO_REDUCTION=$FOLDER2/no_reduction.txt

rm -rf $FILE_SPREAD_REDUCTION
rm -rf $FILE_SPREAD_NO_REDUCTION

threads=32

for N in {10..90..10} {100..200..25} {250..1250..50};
do  
    echo -n $threads " " >> $FILE_SPREAD_REDUCTION
    echo -n $threads " " >> $FILE_SPREAD_NO_REDUCTION
    OMP_NUM_THREADS=$threads OMP_SCHEDULE=static OMP_PROC_BIND=spread OMP_PLACES=cores ./${EXECUTEFOLDER}cpu_reduction $N $ITER $TOLERANCE $START_T >> $FILE_SPREAD_REDUCTION
    OMP_NUM_THREADS=$threads OMP_SCHEDULE=static OMP_PROC_BIND=spread OMP_PLACES=cores ./${EXECUTEFOLDER}cpu_no_reduction $N $ITER $TOLERANCE $START_T >> $FILE_SPREAD_NO_REDUCTION

done
exit 0

## lscpu

for threads in $THREADS;
do
    
    FILE_REDUCTION=$FOLDER/reduction_threads_$threads.txt
    FILE_NO_REDUCTION=$FOLDER/no_reduction_threads_$threads.txt

    rm -rf $FILE_REDUCTION
    rm -rf $FILE_NO_REDUCTION

    ##for N in {10..500..10} {520..600..20} {630..690..30};
    for N in {10..90..10} {100..200..25} {250..1250..50};
    do  
        echo -n $threads " " >> $FILE_REDUCTION
        echo -n $threads " " >> $FILE_NO_REDUCTION
        OMP_NUM_THREADS=$threads OMP_SCHEDULE=static OMP_PROC_BIND=close OMP_PLACES=cores ./${EXECUTEFOLDER}cpu_reduction $N $ITER $TOLERANCE $START_T >> $FILE_REDUCTION
        OMP_NUM_THREADS=$threads OMP_SCHEDULE=static OMP_PROC_BIND=close OMP_PLACES=cores ./${EXECUTEFOLDER}cpu_no_reduction $N $ITER $TOLERANCE $START_T >> $FILE_NO_REDUCTION
    
    done
done



