#!/bin/bash
#SBATCH --job-name=jacobiMPI
#SBATCH --output=jacobiMPI-%j.out
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --mem=128000
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpu-freq=performance
#SBATCH--partition=shortrun
#SBATCH--nodelist=node05

module load cuda
module load openmpi/1.10.4
module load devtoolset-11

GLX=8192
GLY=8192
MAXITER=10000
DUMPSTEP=200
DUMP=0

if [[ ! -d video ]]; then
    echo "creating directory video ..."
    mkdir video
fi

make GLX=$GLX GLY=$GLY MAXITER=$MAXITER DUMP=$DUMP DUMPSTEP=$DUMPSTEP jacobi-mpi

if [[ $? != 0 ]]; then
    echo "[ERROR] compilation FAILED !"
    exit -1
fi


echo "jacobi-mpi ... with 16p"
srun jacobi-mpi

# Attenzione se setto ntasks=16 & ntasks-per-node=1  significa che voglio esegure un processo su 16 nodi macchina => non ci sono in coka
# ntasks=16 & ntasks-per-node=16 invece significa che per un solo nodo richiedo 16 processi, possibile