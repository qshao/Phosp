#!/usr/bin/env bash
# Stage 1-2 only: fetch structure, apply the phospho patch, build topology,
# solvate, ionize, and generate the minimization/nvt/npt/production.mdp files.
# No GPU, no long-running mdrun -- run this first to review the exact MD
# parameters before launching run_ascl1_gpu.sh.
#
# After this completes, review:
#   pS93/output/stage2/{minimization,nvt,npt,production}.mdp
#   pT233/output/stage2/{minimization,nvt,npt,production}.mdp

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH=/data/qsh226/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=phosp-md
export GMXLIB=/data1/qsh226/gmx_forcefields

source "$CONDA_SH"
conda activate "$CONDA_ENV"

for name in pS93 pT233; do
    dir="$SCRIPT_DIR/$name"
    echo "[$name] validating config..."
    (cd "$dir" && phosp validate config.yaml)
    echo "[$name] running stage1-2 (structure prep + mdp generation)..."
    (cd "$dir" && phosp run config.yaml --stages 1,2 --log-level INFO --log-file "$SCRIPT_DIR/${name}_prepare.log")
    echo "[$name] done -> $dir/output/stage2/"
done
