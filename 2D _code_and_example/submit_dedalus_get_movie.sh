#!/bin/bash

#SBATCH --account=def-cumming    # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=0:30:00          # total walltime of your job
#SBATCH --ntasks=64             # number of tasks/processes to run
#SBATCH --mem-per-cpu=4G         # memory per process

# Run on cores across the system : https://docs.alliancecan.ca/wiki/Advanced_MPI_scheduling#Few_cores,_any_number_of_nodes

# Load modules dependencies.
set -euo pipefail
module load StdEnv/2023 gcc openmpi mpi4py/3.1.4 fftw-mpi/3.3.10 hdf5-mpi/1.14.2 python/3.11
module load ffmpeg                      # <-- ensure ffmpeg is available

cd "$SLURM_SUBMIT_DIR"                  # <-- run where snapshots will be written

# Create & activate the venv on all nodes (unchanged from your script) …
srun --ntasks "$SLURM_NNODES" --tasks-per-node=1 bash << 'EOF'
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r dedalus-3.0.2-requirements.txt
EOF

# Activate on the main node
source "$SLURM_TMPDIR/env/bin/activate"
export OMP_NUM_THREADS=1

#---get data---

set +e
srun -n "$SLURM_NTASKS" python3 rotating_ns_flux.py
set -e

#---plot frames + movie---
bash ./making_sphere_from_frames_test_modified.sh

