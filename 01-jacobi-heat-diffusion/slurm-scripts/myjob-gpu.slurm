#!/bin/bash
#SBATCH --job-name=jacobiCUDA
#SBATCH --output=jacobiCUDA%j.out
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --mem=128000
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpu-freq=performance
#SBATCH --exclusive
#SBATCH--partition=skyvolta
#SBATCH--nodelist=node06
#SBATCH--gres=gpu:v100:1

module load nvhpc/24.7
module load devtoolset-11
module load gnuplot/5.2.6

GLX=2048
GLY=2048
MAXITER=10000
DUMPSTEP=200
DUMP=0


make GLX=$GLX GLY=$GLY MAXITER=$MAXITER DUMP=$DUMP DUMPSTEP=$DUMPSTEP jacobi-omp-target
#make GLX=$GLX GLY=$GLY MAXITER=$MAXITER DUMP=$DUMP DUMPSTEP=$DUMPSTEP jacobi-cuda

if [[ $? != 0 ]]; then
    echo "[ERROR] compilation FAILED !"
    exit -1
fi

unset CUDA_VISIBLE_DEVICES

echo "Running jacobi-omp-target..."
 ./jacobi-omp-target

#echo "Running jacobi-cuda..."
# ./jacobi-cuda

if [[ $? != 0 ]]; then 
    echo "[ERROR] run FAILED !"
    exit -1
fi



