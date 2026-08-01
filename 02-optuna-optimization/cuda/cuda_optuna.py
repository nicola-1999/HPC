import subprocess
import optuna
import re
import sys
import os

GLX = 1024
GLY = 1024
MAXITER = 10000
DUMP = 0
DUMPSTEP = 200
USE_DEBUG = 0

# --- La funzione obiettivo per Optuna ---
def objective(trial):
    """
    Funzione obiettivo per Optuna che minimizza il tempo di esecuzione del programma C++.
    """
    # # Definiamo lo spazio di ricerca per i parametri indipendenti:
    # # nthreads (THREADS_PER_BLOCK) e dimgrid.
    # # suggest int con log = True da troppe combinazioni (es. 116) , bisogna controllare che il max num thread sia < 1024
    # valid_combinations = []
    # for tx in [32, 64, 128, 256, 512, 1024]:
    #     for ty in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    #         if tx * ty <= 1024:
    #             valid_combinations.append((tx, ty))
    
    # # VINCOLO: nthreads_x * nthreads_y <= 1024
    # # if nthreads_x * nthreads_y > 1024:
    # #     raise optuna.TrialPruned()
    #     # Optuna sceglie una combinazione valida
    # thread_config = trial.suggest_categorical('thread_config', valid_combinations)
    # nthreads_x, nthreads_y = thread_config

    # num_blocks_x = trial.suggest_categorical('num_blocks_x', [32, 64, 128, 256, 512, 1024, 2048])
    # num_blocks_y = trial.suggest_categorical('num_blocks_y', [32, 64, 128, 256, 512, 1024, 2048])
    # Definiamo i parametri separatamente invece di usare tuple
    # Possibili valori per thread x e y
    valid_tx = [32, 64, 128, 256, 512, 1024]
    valid_ty = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    # consente di testare le coppie speculari
    swapped = trial.suggest_categorical('swapped', [True, False])
    
    # Suggerisci i parametri separatamente
    nthreads_x = trial.suggest_categorical('nthreads_x', valid_tx)
    nthreads_y = trial.suggest_categorical('nthreads_y', valid_ty)


    if swapped:
        temp = nthreads_x
        nthreads_x = nthreads_y
        nthreads_y = temp
    
    # Vincolo: nthreads_x * nthreads_y <= 1024
    if nthreads_x * nthreads_y > 1024:
        raise optuna.TrialPruned()

    num_blocks_x = trial.suggest_categorical('num_blocks_x', [32, 64, 128, 256, 512, 1024, 2048])
    num_blocks_y = trial.suggest_categorical('num_blocks_y', [32, 64, 128, 256, 512, 1024, 2048])
    # --- Passo 1 : compilare 
    try:
        print(f"Compiling and running on cuda device with nthreads_x={nthreads_x}, nthreads_y={nthreads_y}, num_blocks_x={num_blocks_x}, num_blocks_y={num_blocks_y}")

        # Costruiamo il comando completo da eseguire sulla macchina remota (solo per amd)
        #remote_command = (
        #    f"hipcc -O3 {REMOTE_CPP_PATH} -o jacobi-hip "
        #    f"-DNT={nthreads} -DN={num_blocks} -fopenmp -lm && ./jacobi-hip"
        #)

        command = (
            f"rm -f jacobi-cuda-stride && "
            f"nvcc -O3 -gencode arch=compute_70,code=sm_70 "
            f"-DGLX={GLX} -DGLY={GLY} -DMAXITER={MAXITER} "
            f"-DDUMP={DUMP} -DDUMPSTEP={DUMPSTEP} -DUSE_DEBUG={USE_DEBUG} "
            f"-DTHREADS_PER_BLOCK_X={nthreads_x} "
            f"-DTHREADS_PER_BLOCK_Y={nthreads_y} "
            f"-DDIM_GRID_X={num_blocks_x} "
            f"-DDIM_GRID_Y={num_blocks_y} "
            f"-Xcompiler -Wall,-Ofast,-fopenmp "
            f"-lm -o jacobi-cuda-stride jacobi-cuda-stride.cu"
)
        
        compile_result = subprocess.run(
            command,
            shell = True,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if compile_result.returncode != 0:
            print(f"[ERROR] Compilation FAILED!")
            print(f"STDERR: {compile_result.stderr}")
            raise optuna.TrialPruned()
        
        print("[INFO] Compilation successful")

        # --- Passo 2: Eseguire il programma ---
        env = os.environ.copy()
        if 'CUDA_VISIBLE_DEVICES' in env:
            del env['CUDA_VISIBLE_DEVICES']
        
        print("[INFO] Running jacobi-cuda-stride...")
        
        run_result = subprocess.run(
            ['./jacobi-cuda-stride'],
            capture_output=True,
            text=True,
            timeout=600,
            env=env
        )
        
        if run_result.returncode != 0:
            print(f"[ERROR] Run FAILED!")
            print(f"STDERR: {run_result.stderr}")
            raise optuna.TrialPruned()
        
        # --- Passo 3: Analizzare l'output per ottenere il tempo di esecuzione ---
        output_lines = run_result.stdout.strip().split('\n')
        if not output_lines:
            raise ValueError("Program output is empty.")

        last_line = output_lines[-1]
        
        # Usiamo un'espressione regolare specifica per trovare il tempo dopo 'dT:'
        match = re.search(r'dT: (\d+\.?\d*)', last_line)
        if match:
            # Estraiamo il numero dal primo gruppo catturato
            execution_time = float(match.group(1))
        else:
            raise ValueError(f"Could not find a parsable time value in: {last_line}")
        
        print(f"Execution time: {execution_time} msec")
        return execution_time

    except (subprocess.TimeoutExpired, ValueError, Exception) as e: 
        print(f"[ERROR] Trial failed: {e}")
        raise optuna.TrialPruned()
# --- Passo 4: Creare ed eseguire lo studio ---
if __name__ == '__main__':
    # Creazione di un Optuna study per minimizzare il tempo di esecuzione.
    study = optuna.create_study(
        direction='minimize',
        study_name='jacobi_cuda_optimization',
        sampler=optuna.samplers.TPESampler(seed=42)
        )
    
    # Eseguiamo l'ottimizzazione per un certo numero di tentativi.
    print("Starting optimization...")
    study.optimize(objective, n_trials=800)
    
    # Stampiamo i risultati del miglior tentativo.
    print("\nOptimization finished.")
    print("Best trial:")
    print(f"  Execution time: {study.best_value}")
    print("  Params: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
        
# Node 6 non disponibile ...
#SBATCH--partition=skyvolta
#SBATCH--nodelist=node06
#SBATCH--gres=gpu:v100:1