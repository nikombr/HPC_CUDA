#!/bin/bash
#BSUB -J cache_test # name
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

ARCH=`uname -m`

make clean
make realclean
make

if [[ "$ARCH" == "aarch64" ]]
then
    CPU="gracy"
    THREADS="16 32 64 72"
    echo "Running on Gracy :)"
else
    if [[ "$ARCH" == "x86_64" ]]
    then
        CPU="gpuh100"
        THREADS="16 32"
        echo "Running on gpuh100"
    else
        echo "Confused!"
        exit 1
    fi
fi

FOLDER="../results/cpu/$CPU/cache_test"

## lscpu

for threads in $THREADS;
do
    
    FILE_REDUCTION=$FOLDER/reduction_threads_$threads.txt
    FILE_NO_REDUCTION=$FOLDER/no_reduction_threads_$threads.txt

    rm -rf $FILE_REDUCTION
    rm -rf $FILE_NO_REDUCTION

    for N in {10..430..20};
    do  
        echo -n $threads " " >> $FILE_REDUCTION
        echo -n $threads " " >> $FILE_NO_REDUCTION
        OMP_NUM_THREADS=$threads OMP_SCHEDULE=static OMP_PROC_BIND=close OMP_PLACES=cores ./jacobi_reduction $N $ITER $TOLERANCE $START_T >> $FILE_REDUCTION
        OMP_NUM_THREADS=$threads OMP_SCHEDULE=static OMP_PROC_BIND=close OMP_PLACES=cores ./jacobi_no_reduction $N $ITER $TOLERANCE $START_T >> $FILE_NO_REDUCTION
    
    done
done
exit 0

