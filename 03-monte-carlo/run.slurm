#!/bin/bash
#SBATCH --output=montecarlo-pi-%j.out
#SBATCH --partition=shortrun
#SBATCH --ntasks=2

module load cuda
module load openmpi

make mpi 
srun ./montecarlo-mpi-pi

echo " "

sacct -j $SLURM_JOB_ID.0 --format=JobID%16,AllocNodes,NTasks,Elapsed,ConsumedEnergy


echo " "
echo " "
