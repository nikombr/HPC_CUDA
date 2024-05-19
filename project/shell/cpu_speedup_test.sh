#!/bin/bash
#BSUB -J cpu_speedup_test # name
#BSUB -o outfiles/cpu_speedup_test_%J.out # output file
#BSUB -q gpuh100
#BSUB -n 32 ## cores
#BSUB -R "rusage[mem=1GB]" 
#BSUB -W 60 # useable time in minutes
##BSUB -N # send mail when done
#BSUB -R "span[hosts=1]"

ITER=2000
TOLERANCE=-1
START_T=5

## lscpu

ARCH=`uname -m`

## make clean
## make realclean
## make

if [[ "$ARCH" == "aarch64" ]]
then
    CPU="gracy"
    THREADS=$(seq 1 1 72)
    echo "Running on gracy :)"
else
    if [[ "$ARCH" == "x86_64" ]]
    then
        CPU="gpuh100"
        THREADS=$(seq 1 1 32)
        echo "Running on gpuh100"
    else
        echo "Confused!"
        exit 1
    fi
fi

FOLDER="../results/cpu/${CPU}/speedup_test"
EXECUTEFOLDER="../implementation/execute/"

for N in {50..200..50};
do
    
    FILE_REDUCTION=$FOLDER/reduction_$N.txt
    FILE_NO_REDUCTION=$FOLDER/no_reduction_$N.txt

    rm -rf $FILE_REDUCTION
    rm -rf $FILE_NO_REDUCTION

    for threads in $THREADS;
    do  
        echo -n $threads " " >> $FILE_REDUCTION
        echo -n $threads " " >> $FILE_NO_REDUCTION
        OMP_NUM_THREADS=$threads OMP_SCHEDULE=static OMP_PROC_BIND=close OMP_PLACES=cores ./${EXECUTEFOLDER}cpu_reduction $N $ITER $TOLERANCE $START_T >> $FILE_REDUCTION
        OMP_NUM_THREADS=$threads OMP_SCHEDULE=static OMP_PROC_BIND=close OMP_PLACES=cores ./${EXECUTEFOLDER}cpu_no_reduction $N $ITER $TOLERANCE $START_T >> $FILE_NO_REDUCTION

    done
done
exit 0

