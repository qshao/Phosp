#!/usr/bin/env bash
# Research-scale re-run of the PHGDH pT78 benchmark on both GPUs of this node:
#   GPU 0 -> phosphorylated (pT78)   -> phospho/output/
#   GPU 1 -> wild-type reference     -> wt_reference/output_reference/
#
# Protocol: 20 ns NVT, 50 ns NPT, 200 ns production (100 ps save interval),
# using the "globular_protein_research" preset (phosp/protocols/globular_protein_research.yaml).
# At ~100 ns/day observed on this hardware, expect each run to take roughly
# 20/100 + 50/100 + 200/100 =~ 2.7 days. Both GPUs run in parallel, so total
# wall-clock is also ~2.7 days, not double.
#
# All output goes under this directory (benchmarks/phgdh_pT78_200ns/), so the
# earlier benchmarks/phgdh_pT78/ and benchmarks/phgdh_pT78_gpu_rerun/ (1 ns
# quick runs) are never touched.
#
# Requires:
#   - conda env "phosp-md" (phosp + pdb2pqr installed, python 3.10)
#   - a CUDA-enabled gmx binary (see GMX var below — currently the ulab222
#     GROMACS 2022.3 build; see note above the GMX= line for why)
#   - GMXLIB pointing at a writable copy of the CHARMM36m-jul2022 force field
#
# Usage: ./run_phgdh_gpu_200ns.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH=/data/qsh226/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=phosp-md
export GMXLIB=/data1/qsh226/gmx_forcefields
# Using the ulab222 GROMACS 2022.3 CUDA build, not gromacs-2026.3: the 2026.3
# build segfaulted in libgromacs.so partway through a real 20ns NVT run on
# GPU 1 — not trusted for an unattended multi-day job until root-caused.
GMX=/data/ulab222/gromacs-2022.3/bin/gmx_gpu

# Each run pins to one GPU already (gpu_id in its config); cap CPU threads
# per run so two concurrent mdrun processes don't oversubscribe the 40 cores.
export OMP_NUM_THREADS=16

# stage3 (minimization+NVT+NPT+production) has been observed to segfault
# early and rarely (isolated to GPU 1 in testing, not reproducible with a
# controlled restart — looks like a low-probability fault, not a
# deterministic GPU/hardware defect). phosp's checkpoint only tracks whole
# stages, so a retry redoes all of stage3 from scratch; costly if it crashes
# late, but this is the only safety net available without deeper
# checkpoint-resume support in phosp itself.
MAX_RETRIES=3

source "$CONDA_SH"
conda activate "$CONDA_ENV"

# run_one <name> <config_dir> <extra phosp-run flags...>
run_one() {
    local name="$1" dir="$2"; shift 2
    local log="$SCRIPT_DIR/${name}.log"
    echo "[$name] validating config..."
    (cd "$dir" && phosp validate config.yaml) || { echo "[$name] validate FAILED"; return 1; }

    echo "[$name] running stage1-3 (structure prep + GPU MD: 20ns NVT / 50ns NPT / 200ns production)..."
    local attempt=1
    while true; do
        if (cd "$dir" && phosp run config.yaml --stages 1,2,3 "$@" --log-level INFO --log-file "$log"); then
            break
        fi
        echo "[$name] stage1-3 attempt $attempt/$MAX_RETRIES FAILED"
        attempt=$((attempt + 1))
        if [[ $attempt -gt $MAX_RETRIES ]]; then
            echo "[$name] all $MAX_RETRIES attempts failed. Giving up."
            return 1
        fi
        echo "[$name] retrying in 30s..."
        sleep 30
    done

    # phosp's stage4 analysis (RMSD/RMSF/Rg/...) reads production.xtc directly
    # with no PBC treatment. A protein can wrap across the periodic box during
    # a run, which corrupts alignment-based metrics like RMSD. Re-center/
    # rewrap with trjconv before analysis, in place, so stage4 picks it up.
    local out_root="$dir/output"
    [[ "$*" == *--reference* ]] && out_root="$dir/output_reference"
    local prod_dir="$out_root/stage3/production"
    echo "[$name] PBC-correcting production trajectory..."
    mv "$prod_dir/production.xtc" "$prod_dir/production_raw.xtc"
    printf "1\n0\n" | "$GMX" trjconv \
        -s "$prod_dir/production.tpr" \
        -f "$prod_dir/production_raw.xtc" \
        -o "$prod_dir/production.xtc" \
        -pbc mol -center -ur compact >> "$log" 2>&1

    echo "[$name] running stage4 (analysis)..."
    (cd "$dir" && phosp run config.yaml --stages 4 "$@" --log-level INFO --log-file "$log")
    echo "[$name] done -> $out_root/stage4/report.html"
}

echo "=== Launching phospho (GPU 0) and wt_reference (GPU 1) in parallel ==="
echo "=== Expect ~2.7 days per run (20ns NVT + 50ns NPT + 200ns production @ ~100 ns/day) ==="
run_one "phospho"      "$SCRIPT_DIR/phospho"                  > "$SCRIPT_DIR/phospho.stdout"      2>&1 &
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
