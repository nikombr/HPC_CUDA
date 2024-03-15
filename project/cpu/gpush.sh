#!/bin/bash
#BSUB -J cpu # name
#BSUB -o outfiles/close_%J.out # output file
#BSUB -q hpcintrogpu
#BSUB -n 32 ## cores
#BSUB -R "rusage[mem=1GB]" 
#BSUB -W 60 # useable time in minutes
##BSUB -N # send mail when done
#BSUB -R "span[hosts=1]"

N=200 ## 55, 165, 275, 385
ITER=2000 ## 200000, 8000, 1500, 600
TOLERANCE=-1
START_T=5

lscpu

FOLDER="../results/cpu/gpush"
FILE_REDUCTION=$FOLDER/reduction.txt
FILE_NO_REDUCTION=$FOLDER/no_reduction.txt

rm -rf $FILE_REDUCTION
rm -rf $FILE_NO_REDUCTION

for threads in {1..32..1};
do  
    echo -n $threads " " >> $FILE_REDUCTION
    echo -n $threads " " >> $FILE_NO_REDUCTION
    OMP_NUM_THREADS=$threads OMP_SCHEDULE=static OMP_PROC_BIND=close OMP_PLACES=cores ./jacobi_reduction $N $ITER $TOLERANCE $START_T >> $FILE_REDUCTION
    OMP_NUM_THREADS=$threads OMP_SCHEDULE=static OMP_PROC_BIND=close OMP_PLACES=cores ./jacobi_no_reduction $N $ITER $TOLERANCE $START_T >> $FILE_NO_REDUCTION

done

exit 0

