#!/bin/bash
#BSUB -J gpuh100_cache_test # name
#BSUB -o outfiles/close_%J.out # output file
#BSUB -q gpuh100
#BSUB -n 32 ## cores
#BSUB -R "rusage[mem=1GB]" 
#BSUB -W 60 # useable time in minutes
##BSUB -N # send mail when done
#BSUB -R "span[hosts=1]"

ITER=2000
TOLERANCE=-1
START_T=5

FOLDER="../results/cpu/gpuh100/cache_test"

lscpu

THREADS="16 32"

for threads in $THREADS;
do
    
    FILE_REDUCTION=$FOLDER/reduction_threads_$threads.txt
    FILE_NO_REDUCTION=$FOLDER/no_reduction_threads_$threads.txt

    rm -rf $FILE_REDUCTION
    rm -rf $FILE_NO_REDUCTION

    for N in {20..300..20};
    do  
        echo -n $threads " " >> $FILE_REDUCTION
        echo -n $threads " " >> $FILE_NO_REDUCTION
        OMP_NUM_THREADS=$threads OMP_SCHEDULE=static OMP_PROC_BIND=close OMP_PLACES=cores ./jacobi_reduction $N $ITER $TOLERANCE $START_T >> $FILE_REDUCTION
        OMP_NUM_THREADS=$threads OMP_SCHEDULE=static OMP_PROC_BIND=close OMP_PLACES=cores ./jacobi_no_reduction $N $ITER $TOLERANCE $START_T >> $FILE_NO_REDUCTION

    done
done
exit 0

