import subprocess
import optuna
import re
import sys
from datetime import datetime

GLX = 1024
GLY = 1024
MAXITER = 10000
DUMP = 0
DUMPSTEP = 200
# GPU_ARCH = "gfx90a" 

SSH_USER = "nmischia"
SSH_HOST = "cassone"

# Percorso completo allo script C++ sulla macchina remota.
REMOTE_CPP_PATH = "/home/nmischia/projectWork/amd/jacobi-amd.cpp"

# --- La funzione obiettivo per Optuna ---
def objective(trial):
    """
    Funzione obiettivo per Optuna che minimizza il tempo di esecuzione del programma C++.
    """
    # Definiamo i parametri separatamente invece di usare tuple
    # Possibili valori per thread x e y
    valid_tx = [32, 64, 128, 256, 512, 1024]
    valid_ty = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    
    # consente di testare le coppie speculari
    swapped = trial.suggest_categorical('swapped', [True, False])

    nthreads_x = trial.suggest_categorical('nthreads_x', valid_tx)
    nthreads_y = trial.suggest_categorical('nthreads_y', valid_ty)

    if swapped:
        temp = nthreads_x
        nthreads_x = nthreads_y
        nthreads_y = temp

    # Vincolo: nthreads_x * nthreads_y <= 1024
    if nthreads_x * nthreads_y > 1024:
        raise optuna.TrialPruned()

    num_blocks_x = trial.suggest_categorical('num_blocks_x', [8,16, 32, 64, 128, 256, 512, 1024, 2048])
    num_blocks_y = trial.suggest_categorical('num_blocks_y', [8,16, 32, 64, 128, 256, 512, 1024, 2048])
    
    # --- Passo 1 & 2: Compilare ed eseguire tramite SSH per connettersi a cassone ---
    try:
        print(f"Trial: nthreads_x={nthreads_x}, nthreads_y={nthreads_y}, "
              f"num_blocks_x={num_blocks_x}, num_blocks_y={num_blocks_y}")

        # Prima verificare che il file esista
        check_file_command = f"test -f {REMOTE_CPP_PATH} && echo 'EXISTS' || echo 'NOT_FOUND'"
        check_result = subprocess.run(
            ['ssh', f"{SSH_USER}@{SSH_HOST}", check_file_command],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if 'NOT_FOUND' in check_result.stdout:
            print(f"ERROR: File {REMOTE_CPP_PATH} not found on remote host!", file=sys.stderr)
            print("Please check the file path on the remote machine.", file=sys.stderr)
            raise optuna.exceptions.TrialPruned()

        # Costruire il comando completo da eseguire sulla macchina remota
        remote_command = (
            f"rm -f jacobi-hip && "
            f"hipcc -O3 {REMOTE_CPP_PATH} -o jacobi-hip "
            f"-DGLX={GLX} -DGLY={GLY} -DMAXITER={MAXITER} "
            f"-DTHREADGRIDX={nthreads_x} "
            f"-DTHREADGRIDY={nthreads_y} "
            f"-DBLOCKGRIDX={num_blocks_x} "
            f"-DBLOCKGRIDY={num_blocks_y} "  
            f"-fopenmp -lm && ./jacobi-hip"
        )
        # remote_command = (
        #     f"hipcc -O3 {REMOTE_CPP_PATH} -o jacobi-hip "
        #     f"-DGLX={GLX} -DGLY={GLY} -DMAXITER={MAXITER} "
        #     f"-DDUMP={DUMP} -DDUMPSTEP={DUMPSTEP} "
        #     f"-DTHREADGRIDX={nthreads_x} "
        #     f"-DTHREADGRIDY={nthreads_y} "
        #     f"-DBLOCKGRIDX={num_blocks_x} "
        #     f"-DBLOCKGRIDY={num_blocks_y} "
        #     f"-Wall,-Ofast,-fopenmp "
        #     f"-lm && ./jacobi-hip"
        # )
        
        # Esegui il comando SSH con timeout
        result = subprocess.run(
            ['ssh', f"{SSH_USER}@{SSH_HOST}", remote_command],
            check=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minuti di timeout
        )
        
        # --- Passo 3: Analizzare l'output per ottenere il tempo di esecuzione ---
        output_lines = result.stdout.strip().split('\n')
        if not output_lines:
            raise ValueError("Program output is empty.")

        last_line = output_lines[-1]
        
        # Usiamo un'espressione regolare specifica per trovare il tempo dopo 'dT:'
        match = re.search(r'dT:\s*(\d+\.?\d*)', last_line)
        if match:
            execution_time = float(match.group(1))
        else:
            # Ricerca in tutte le righe
            for line in reversed(output_lines):
                match = re.search(r'dT:\s*(\d+\.?\d*)', line)
                if match:
                    execution_time = float(match.group(1))
                    break
            else:
                print(f"Output received:\n{result.stdout}", file=sys.stderr)
                raise ValueError(f"Could not find a parsable time value in output")
        
        print(f"✓ Execution time: {execution_time} msec")
        return execution_time

    except subprocess.TimeoutExpired:
        print(f"Execution timed out after 300 seconds", file=sys.stderr)
        raise optuna.exceptions.TrialPruned()
    except subprocess.CalledProcessError as e:
        print(f"SSH command failed with return code {e.returncode}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        print(f"stdout: {e.stdout}", file=sys.stderr)
        raise optuna.exceptions.TrialPruned()
    except ValueError as e:
        print(f"Error parsing output: {e}", file=sys.stderr)
        raise optuna.exceptions.TrialPruned()
    except FileNotFoundError as e:
        print(f"SSH command not found: {e}. Is SSH installed and in your PATH?", file=sys.stderr)
        raise optuna.exceptions.TrialPruned()
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        raise optuna.exceptions.TrialPruned()

# --- Passo 4: Creare ed eseguire lo studio ---
if __name__ == '__main__':
    # Nome del file di output
    output_filename = f"optuna_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.out"
    
    # Crea uno studio Optuna per minimizzare il tempo di esecuzione.
    study = optuna.create_study(
        direction='minimize',
        study_name='jacobi_amd_optimization',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    print("Starting optimization...")
    print(f"Target: Optimize Jacobi solver on AMD GPU with {GLX}x{GLY} grid")
    print(f"Remote host: {SSH_USER}@{SSH_HOST}")
    print(f"Remote file: {REMOTE_CPP_PATH}")
    print(f"Output file: {output_filename}\n")
    
    try:
        study.optimize(objective, n_trials=800)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    
    # Stampa del miglior tentativo.
    results_text = []
    results_text.append("=" * 60)
    results_text.append("OPTUNA OPTIMIZATION RESULTS - JACOBI AMD")
    results_text.append("=" * 60)
    results_text.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results_text.append(f"Grid size: {GLX} x {GLY}")
    results_text.append(f"Max iterations: {MAXITER}")
    results_text.append(f"Remote host: {SSH_USER}@{SSH_HOST}")
    results_text.append(f"Remote file: {REMOTE_CPP_PATH}")
    results_text.append("")
    
    # Verificare che ci siano trial completati
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if completed_trials:
        results_text.append(f"Completed trials: {len(completed_trials)}/{len(study.trials)}")
        results_text.append("")
        results_text.append("BEST TRIAL:")
        results_text.append("-" * 60)
        results_text.append(f"Execution time: {study.best_value:.2f} msec")
        results_text.append("")
        results_text.append("Parameters:")
        for key, value in study.best_params.items():
            results_text.append(f"  {key}: {value}")
        
        # Calcolare il prodotto totale dei thread
        best_threads_total = study.best_params['nthreads_x'] * study.best_params['nthreads_y']
        results_text.append("")
        results_text.append(f"Total threads per block: {best_threads_total}")
        results_text.append(f"Total blocks: {study.best_params['num_blocks_x']} x {study.best_params['num_blocks_y']}")
        results_text.append(f"Total threads: {best_threads_total * study.best_params['num_blocks_x'] * study.best_params['num_blocks_y']}")
        
        # Aggiungi statistiche sui top 5 trial
        # results_text.append("")
        # results_text.append("=" * 60)
        # results_text.append("TOP 5 TRIALS:")
        # results_text.append("=" * 60)
        # sorted_trials = sorted(completed_trials, key=lambda t: t.value)[:5]
        # for i, trial in enumerate(sorted_trials, 1):
        #     results_text.append(f"\n{i}. Trial #{trial.number}")
        #     results_text.append(f"   Time: {trial.value:.2f} msec")
        #     results_text.append(f"   nthreads_x: {trial.params['nthreads_x']}, nthreads_y: {trial.params['nthreads_y']}")
        #     results_text.append(f"   num_blocks_x: {trial.params['num_blocks_x']}, num_blocks_y: {trial.params['num_blocks_y']}")
    else:
        results_text.append("⚠ WARNING: No trials completed successfully!")
        results_text.append(f"Total trials attempted: {len(study.trials)}")
        results_text.append("All trials were pruned due to errors.")
        results_text.append("")
        results_text.append("Please check:")
        results_text.append(f"1. File exists at: {REMOTE_CPP_PATH}")
        results_text.append(f"2. SSH connection to {SSH_USER}@{SSH_HOST} works")
        results_text.append(f"3. hipcc compiler is available on the remote host")
        results_text.append(f"4. The C++ code compiles and runs correctly")
    
    results_text.append("")
    results_text.append("=" * 60)
    
    # Stampa a schermo
    print("\n" + "\n".join(results_text))
    
    # Scrivi nel file .out
    with open(output_filename, 'w') as f:
        f.write("\n".join(results_text))
    
    print(f"\n✓ Results saved to: {output_filename}")