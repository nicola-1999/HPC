#!/bin/bash
#SBATCH --job-name=jacobi-OMP
#SBATCH --output=jacobi-OMP-%j.out
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --mem=128000
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --cpu-freq=performance

module load devtoolset-11
module load gnuplot/5.2.6

GLX=2048
GLY=2048
MAXITER=10000
DUMPSTEP=200
DUMP=0

if [[ ! -d video ]]; then
    echo "creating directory video ..."
    mkdir video
fi

make CC=gcc GLX=$GLX GLY=$GLY MAXITER=$MAXITER DUMP=$DUMP DUMPSTEP=$DUMPSTEP jacobi-omp
#make CC=gcc GLX=$GLX GLY=$GLY MAXITER=$MAXITER DUMP=$DUMP DUMPSTEP=$DUMPSTEP jacobi-serial

if [[ $? != 0 ]]; then
    echo "[ERROR] compilation FAILED !"
    exit -1
fi

#echo "Running jacobi-serial..."
#./jacobi-serial

echo "Running jacobi-omp..."
export OMP_PLACES=cores # place threads on hardware cores
export OMP_PROC_BIND=close # place threads close to each other
./jacobi-omp

if [[ $? != 0 ]]; then 
    echo "[ERROR] run FAILED !"
    exit -1
fi

exit 0