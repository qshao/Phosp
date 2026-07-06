#!/usr/bin/env bash
# Run the two ASCL1 (UniProt P50553) phosphorylation cases on both GPUs:
#   GPU 0 -> pS93/config.yaml  (pSer93)  -> pS93/output/
#   GPU 1 -> pT233/config.yaml (pThr233) -> pT233/output/
#
# Protocol: phosp/protocols/globular_protein_100ns_eq.yaml
#   100 ns NVT (heavy-atom restraints) + 100 ns NPT (backbone restraints)
#   + 300 ns production, 1.0 nm vdW/Coulomb cutoff.
#
# Usage:
#   ./prepare.sh          # stages 1-2 only: structure prep + mdp generation (fast, no GPU)
#                          # -> review pS93/output/stage2/{minimization,nvt,npt,production}.mdp
#   ./run_ascl1_gpu.sh     # stages 3-4: GPU production MD + analysis (long-running)
#
# Requires (see benchmarks/phgdh_pT78_gpu_rerun/README.md for how these were set up):
#   - conda env "phosp-md" (phosp + pdb2pqr installed, python 3.10)
#   - a CUDA-enabled gmx binary (referenced by gromacs.binary in the configs)
#   - GMXLIB pointing at a writable copy of the CHARMM36m-jul2022 force field

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH=/data/qsh226/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=phosp-md
export GMXLIB=/data1/qsh226/gmx_forcefields
GMX_CUDA=/home/AD/qsh226/software/gromacs-2026.3/bin/gmx

# `nvidia-smi topo -m` reports both GPUs' CPU affinity as "0-9,20-29" (socket 0,
# 10 physical cores / 20 hardware threads — the other socket, cores 10-19/30-39,
# is NUMA-remote from both GPUs). Split that local pool evenly and disjointly
# between the two cases (5 physical cores / 10 threads each) and pin each
# mdrun's process affinity with taskset so GROMACS's own thread pinning
# ("-pin auto") has a matched core count to lock onto, instead of both
# processes contending unpinned across all 40 cores.
CPUSET_S=0-4,20-24
CPUSET_T=5-9,25-29
export OMP_NUM_THREADS=10

source "$CONDA_SH"
conda activate "$CONDA_ENV"

# run_one <name> <config_dir> <cpuset>
run_one() {
    local name="$1" dir="$2" cpuset="$3"
    local log="$SCRIPT_DIR/${name}.log"

    echo "[$name] validating config..."
    (cd "$dir" && phosp validate config.yaml) || { echo "[$name] validate FAILED"; return 1; }

    echo "[$name] running stage3 (100ns NVT + 100ns NPT + 300ns production, GPU, CPUs $cpuset)..."
    (cd "$dir" && taskset -c "$cpuset" phosp run config.yaml --stages 3 --log-level INFO --log-file "$log")

    # phosp's stage4 analysis reads production.xtc directly with no PBC
    # treatment. Re-center/rewrap with trjconv before analysis, in place.
    local prod_dir="$dir/output/stage3/production"
    echo "[$name] PBC-correcting production trajectory..."
    mv "$prod_dir/production.xtc" "$prod_dir/production_raw.xtc"
    printf "1\n0\n" | taskset -c "$cpuset" "$GMX_CUDA" trjconv \
        -s "$prod_dir/production.tpr" \
        -f "$prod_dir/production_raw.xtc" \
        -o "$prod_dir/production.xtc" \
        -pbc mol -center -ur compact >> "$log" 2>&1

    echo "[$name] running stage4 (analysis)..."
    (cd "$dir" && taskset -c "$cpuset" phosp run config.yaml --stages 4 --log-level INFO --log-file "$log")
    echo "[$name] done -> $dir/output/stage4/report.html"
}

echo "=== Launching pS93 (GPU 0, CPUs $CPUSET_S) and pT233 (GPU 1, CPUs $CPUSET_T) ==="
run_one "pS93"  "$SCRIPT_DIR/pS93"  "$CPUSET_S" > "$SCRIPT_DIR/pS93.stdout"  2>&1 &
PID_S=$!
# Stagger the second launch slightly — starting both mdrun processes' CUDA
# context init at the exact same instant previously caused one to crash
# silently (no fatal-error text, just a dead process) on this driver/build.
sleep 5
run_one "pT233" "$SCRIPT_DIR/pT233" "$CPUSET_T" > "$SCRIPT_DIR/pT233.stdout" 2>&1 &
PID_T=$!

status=0
wait "$PID_S" || { echo "pS93 run FAILED — see pS93.stdout / pS93.log"; status=1; }
wait "$PID_T" || { echo "pT233 run FAILED — see pT233.stdout / pT233.log"; status=1; }

exit $status
