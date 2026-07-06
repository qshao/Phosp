#!/usr/bin/env bash
# Standalone relauncher for the wt_reference run (GPU 1), with retries.
#
# Context: the wt_reference NVT phase segfaulted twice early on (once with
# GROMACS 2026.3, once with 2022.3), both isolated to GPU 1. A controlled
# diagnostic (minimize -> NVT with the exact same flags, in isolation on
# GPU 1) ran past 5800 steps with no issue, so GPU 1 itself is not reliably
# broken -- the earlier failures look like a rare, low-probability fault
# rather than a deterministic hardware defect. This wrapper retries a few
# times so an unattended recurrence doesn't just kill the run permanently.
#
# phosp's checkpoint only tracks whole-stage completion, so a retry redoes
# all of stage3 (minimization+NVT+NPT+production) from scratch -- costly if
# it crashes late, but the only option without deeper checkpoint-resume
# support in phosp itself.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH=/data/qsh226/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=phosp-md
export GMXLIB=/data1/qsh226/gmx_forcefields
GMX=/data/ulab222/gromacs-2022.3/bin/gmx_gpu
export OMP_NUM_THREADS=16
MAX_RETRIES=3

source "$CONDA_SH"
conda activate "$CONDA_ENV"

DIR="$SCRIPT_DIR/wt_reference"
LOG="$SCRIPT_DIR/wt_reference.log"

attempt=1
while [[ $attempt -le $MAX_RETRIES ]]; do
    echo "[wt_reference] attempt $attempt/$MAX_RETRIES: running stage1-3..."
    if (cd "$DIR" && phosp run config.yaml --stages 1,2,3 --reference --log-level INFO --log-file "$LOG"); then
        echo "[wt_reference] stage1-3 succeeded on attempt $attempt"
        break
    fi
    echo "[wt_reference] attempt $attempt FAILED — see $LOG"
    attempt=$((attempt + 1))
    if [[ $attempt -le $MAX_RETRIES ]]; then
        echo "[wt_reference] retrying in 30s..."
        sleep 30
    else
        echo "[wt_reference] all $MAX_RETRIES attempts failed. Giving up."
        exit 1
    fi
done

# Same PBC-fix as the main orchestrator before running stage4.
prod_dir="$DIR/output_reference/stage3/production"
echo "[wt_reference] PBC-correcting production trajectory..."
mv "$prod_dir/production.xtc" "$prod_dir/production_raw.xtc"
printf "1\n0\n" | "$GMX" trjconv \
    -s "$prod_dir/production.tpr" \
    -f "$prod_dir/production_raw.xtc" \
    -o "$prod_dir/production.xtc" \
    -pbc mol -center -ur compact >> "$LOG" 2>&1

echo "[wt_reference] running stage4 (analysis)..."
(cd "$DIR" && phosp run config.yaml --stages 4 --reference --log-level INFO --log-file "$LOG")
echo "[wt_reference] done -> $DIR/output_reference/stage4/report.html"
