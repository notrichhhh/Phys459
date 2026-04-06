#!/bin/bash

export OMP_NUM_THREADS=1
set -euo pipefail

TASKS=(flux u_theta_mag T y y_dot u_phi_mag)
LABELS=(F u_theta T y ydot u_phi)
NTASKS=${#TASKS[@]}

THETA_MASTER="theta_master.csv"
python3 make_theta_master_csv.py "${TASKS[0]}" snapshots/*.h5 --out "${THETA_MASTER}"

#: << 'DISABLED_BLOCK'
for ((i=0; i<NTASKS; i++)); do
    task="${TASKS[$i]}"
    label="${LABELS[$i]}"

    # ============================================================
    # sphere frames -> CWD (shared), then delete
    # ============================================================
    frames_dir="./frames_${label}"
    movie_file="./movie_${label}.mp4"

    rm -rf "${frames_dir}"
    mkdir -p "${frames_dir}"

    if [[ "$task" == "flux" ]]; then
        linear_flag=""
    else
        linear_flag="--linear"
    fi

    srun -n "${SLURM_NTASKS:-1}" python3 plot_sphere_s2.py \
        "$task" snapshots/*.h5 \
        --output "${frames_dir}" \
        ${linear_flag}

    ffmpeg -y -framerate 48 -pattern_type glob -i "${frames_dir}/write_*.png" \
        -c:v libx264 -pix_fmt yuv420p -crf 18 "${movie_file}"

    rm -rf "${frames_dir}"

    # ============================================================
    # 1D phi=0 frames -> CWD (shared), then delete
    # ============================================================
    phi_csv="./phi0_${label}.csv"
    frames_1d="./frames_${label}_projection"
    movie_1d="./movie_${label}_projection.mp4"

    rm -rf "${frames_1d}"
    mkdir -p "${frames_1d}"

    srun -n "${SLURM_NTASKS:-1}" python3 make_phi0_csv.py "$task" snapshots/*.h5 \
        --phi=0 \
        --theta-master "${THETA_MASTER}" \
        --out "${phi_csv}"

    srun -n "${SLURM_NTASKS:-1}" python3 plot_phi0_data.py "$task" \
        --data "${phi_csv}" \
        --theta "${THETA_MASTER}" \
        --out "${frames_1d}" \
        --movie "${movie_1d}"

    rm -rf "${frames_1d}"
done
#DISABLED_BLOCK

srun -n 1 python3 plot_max_vs_time.py phi0_u_theta.csv
#srun -n 1 python3 plot_sphavg.py phi0_y.csv --out sphavg_ydot.png
srun -n 1 python3 plot_equator_and_sphavg_phi0_y.py

echo "Done."




