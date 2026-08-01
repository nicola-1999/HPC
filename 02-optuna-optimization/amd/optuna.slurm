#!/bin/bash
#SBATCH --job-name=OptunaJacobiCuda
#SBATCH --output=OptunaJacobiCuda-%j.out
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --mem=128000
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpu-freq=performance
#SBATCH --exclusive
#SBATCH--gres=gpu:1

# Carica i moduli necessari
module load nvhpc/24.7
module load devtoolset-11
#module load gnuplot/5.2.6

# Carica Python/Anaconda se necessario
# module load anaconda3
# source activate your_env

# Assicurati che CUDA_VISIBLE_DEVICES non sia settato
unset CUDA_VISIBLE_DEVICES

# Esegui lo script Python con Optuna
echo "Starting Optuna optimization..."
python amd_optuna.py

if [[ $? != 0 ]]; then 
    echo "[ERROR] Optuna optimization FAILED !"
    exit -1
fi

echo "Optimization completed successfully!"