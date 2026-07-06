#!/usr/bin/env bash
# Re-run the PHGDH pT78 benchmark on both GPUs of this node:
#   GPU 0 -> phosphorylated (pT78)   -> phospho/output/
#   GPU 1 -> wild-type reference     -> wt_reference/output_reference/
#
# All output goes under this directory (benchmarks/phgdh_pT78_gpu_rerun/),
# so the existing benchmarks/phgdh_pT78/ case is never touched.
#
# Requires (see README.md in this directory for how these were set up):
#   - conda env "phosp-md" (phosp + pdb2pqr installed, python 3.10)
#   - a CUDA-enabled gmx binary (referenced by gromacs.binary in the configs)
#   - GMXLIB pointing at a writable copy of the CHARMM36m-jul2022 force field
#
# Usage: ./run_phgdh_gpu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH=/data/qsh226/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=phosp-md
export GMXLIB=/data1/qsh226/gmx_forcefields
GMX_CUDA=/home/AD/qsh226/software/gromacs-2026.3/bin/gmx

# Each run pins to one GPU already (gpu_id in its config); cap CPU threads
# per run so two concurrent mdrun processes don't oversubscribe the 40 cores.
export OMP_NUM_THREADS=16

source "$CONDA_SH"
conda activate "$CONDA_ENV"

# run_one <name> <config_dir> <extra phosp-run flags...>
run_one() {
    local name="$1" dir="$2"; shift 2
    local log="$SCRIPT_DIR/${name}.log"
    echo "[$name] validating config..."
    (cd "$dir" && phosp validate config.yaml) || { echo "[$name] validate FAILED"; return 1; }

    echo "[$name] running stage1-3 (structure prep + GPU MD)..."
    (cd "$dir" && phosp run config.yaml --stages 1,2,3 "$@" --log-level INFO --log-file "$log")

    # phosp's stage4 analysis (RMSD/RMSF/Rg/...) reads production.xtc directly
    # with no PBC treatment. A protein can wrap across the periodic box during
    # a run, which corrupts alignment-based metrics like RMSD. Re-center/
    # rewrap with trjconv before analysis, in place, so stage4 picks it up.
    local out_root="$dir/output"
    [[ "$*" == *--reference* ]] && out_root="$dir/output_reference"
    local prod_dir="$out_root/stage3/production"
    echo "[$name] PBC-correcting production trajectory..."
    mv "$prod_dir/production.xtc" "$prod_dir/production_raw.xtc"
    printf "1\n0\n" | "$GMX_CUDA" trjconv \
        -s "$prod_dir/production.tpr" \
        -f "$prod_dir/production_raw.xtc" \
        -o "$prod_dir/production.xtc" \
        -pbc mol -center -ur compact >> "$log" 2>&1

    echo "[$name] running stage4 (analysis)..."
    (cd "$dir" && phosp run config.yaml --stages 4 "$@" --log-level INFO --log-file "$log")
    echo "[$name] done -> $out_root/stage4/report.html"
}

echo "=== Launching phospho (GPU 0) and wt_reference (GPU 1) in parallel ==="
run_one "phospho"     "$SCRIPT_DIR/phospho"                 > "$SCRIPT_DIR/phospho.stdout"     2>&1 &
PID_P=$!
run_one "wt_reference" "$SCRIPT_DIR/wt_reference" --reference > "$SCRIPT_DIR/wt_reference.stdout" 2>&1 &
PID_W=$!

status=0
wait "$PID_P" || { echo "phospho run FAILED — see phospho.stdout / phospho.log"; status=1; }
wait "$PID_W" || { echo "wt_reference run FAILED — see wt_reference.stdout / wt_reference.log"; status=1; }

if [[ $status -eq 0 ]]; then
    echo "=== Both runs complete. Generating comparison plots... ==="
    python "$SCRIPT_DIR/compare.py"
fi

exit $status
