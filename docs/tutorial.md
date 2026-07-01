# Tutorial: phosphorylating ubiquitin and running an MD simulation

This tutorial walks through a complete phosp run from a fresh environment to an HTML analysis report. We use ubiquitin (a small 76-residue protein, PDB: 1UBQ) with phosphorylation at Thr66 as a concrete example. The same steps apply to any protein and any combination of phospho-sites.

Estimated time: 10 minutes to set up. MD run time varies enormously by hardware and system size — a 10 ns production run can take anywhere from ~20 minutes (GPU) to well over a day (CPU-only, larger protein). GROMACS needs a CUDA or ROCm build to use an NVIDIA/AMD GPU; some platforms (e.g. conda-forge's linux-aarch64 GROMACS) only ship an OpenCL build, which silently falls back to CPU-only on GPUs it doesn't support — check your `mdrun` log for a line like `status: incompatible` under "GPU support" if a run seems unexpectedly slow.

---

## Prerequisites

Before starting this tutorial, complete the [installation guide](installation.md). In particular:

- phosp is installed in an active virtual environment
- `gmx --version` prints without error
- `pdb2pqr --version` prints `3.x.x`
- `phosp validate examples/ubiquitin_pThr/run.yaml` prints four green checkmarks

---

## 1. Understand the input

### The protein structure

We use the ubiquitin structure bundled with phosp:

```
tests/fixtures/ubiquitin.pdb
```

Open it in your structure viewer (PyMOL, UCSF ChimeraX, VMD) and note:

- Chain A, 76 residues (Met1 → Gly76)
- Residue 66 is a threonine (THR66)
- No waters or HETATM records (the bundled file is pre-cleaned)

When using your own protein:
- Single-chain proteins work directly. Multi-chain proteins work too — specify the chain ID in the config.
- Remove crystallographic waters, ligands, and HETATM records you do not need before running phosp (or let Stage 1 clean them automatically — phosp uses `_CleanSelect` to strip waters and unknown HETATMs).
- Make sure residue numbers in the PDB file match what you put in the config. If the PDB has insertion codes or missing residues, check the numbering carefully.

### What phosphorylation does

phosp's Stage 1 replaces the residue in-place:

| Original | Phospho-type | New residue | What changes |
|---|---|---|---|
| SER | pSer | SEP | Serine → O-phosphoserine |
| THR | pThr | TPO | Threonine → O-phosphothreonine |
| TYR | pTyr | PTR | Tyrosine → O-phosphotyrosine |

The phosphate group adds a formal charge of −2 at physiological pH. phosp handles the force field parameters automatically through CHARMM36m's native SEP/TPO/PTR definitions.

---

## 2. Create your project directory

```bash
mkdir -p ~/phosp_tutorial
cd ~/phosp_tutorial
```

---

## 3. Generate and edit the config

```bash
phosp init config.yaml
```

This writes a fully annotated starter config. Open it and make the following changes:

```yaml
# ~/phosp_tutorial/config.yaml

input:
  source: pdb
  path: /path/to/Phosp/tests/fixtures/ubiquitin.pdb   # absolute path
  ph: 7.4

modification:
  sites:
    - chain: A
      resid: 66
      resname: THR
      phospho_type: pThr

forcefield: charmm36m
protocol: globular_protein

gromacs:
  binary: gmx          # or "gmx_mpi" if you have an MPI build

simulation:
  production_time_ns: 10.0     # 10 ns is a reasonable starting point
  output_freq_ps: 10.0
  water_model: tip3p
  box_type: dodecahedron
  salt_concentration_mM: 150.0
  gpu_id: ~                    # ~ = nonbonded-only GPU use; set to 0, 1 … for full offload (see below)
  runner: local

analysis:
  plugins:
    - rmsd
    - rmsf
    - radius_of_gyration
    - secondary_structure
    - hbond
    - sasa
  rmsd:
    selection: name CA
  rmsf:
    selection: name CA
  sasa:
    residues: []
```

### Key points

**`input.path`** must be an absolute path or a path relative to where you run `phosp`, not relative to the config file. Using an absolute path avoids confusion.

**`modification.sites`** — `resname` must exactly match what appears in the PDB (column 18–20). If the PDB says `THR`, write `THR`, not `Thr`.

**`production_time_ns: 10.0`** — 10 ns takes about 1–2 h on a modern laptop CPU. For a production research simulation use 100–500 ns. For a quick validation test, set it to `0.001` (1 ps) and use a custom fast protocol (see the [README](../README.md#simulation-protocols)).

---

## 4. Validate before running

Always validate before starting a long simulation:

```bash
phosp validate config.yaml
```

Expected output:

```
  ✓ Config valid
  ✓ gmx found
  ✓ pdb2pqr found
  ✓ Force field ready
```

If anything fails, check the [troubleshooting section](../README.md#troubleshooting).

You can also do a full dry-run that checks input files and estimates disk usage without actually running any stage:

```bash
phosp run config.yaml --dry-run
# Estimated disk space needed: 10.5 GB
# Dry run complete — config and environment OK
```

---

## 5. Run the pipeline

```bash
phosp run config.yaml
```

phosp prints a live progress display as each stage runs. Here is what happens at each stage.

### Stage 1 — Chemical modification (~5 seconds)

1. **Acquire** — copies the PDB (or downloads from AlphaFold/RCSB for UniProt input)
2. **Clean** — removes waters and unknown HETATMs
3. **Protonate** — runs `pdb2pqr` with PROPKA at the configured pH to assign all hydrogen positions and protonation states
4. **Modify** — applies the phospho patch: changes THR66 → TPO, adds force field parameters from the CHARMM36m SEP/TPO/PTR definitions
5. **Write** — saves `modified.pdb` and a `modification_manifest.json` log

**Output files:**
```
output/stage1/
├── input.pdb                     # copy of the original
├── cleaned.pdb                   # after HETATM removal
├── protonated.pdb                # after pdb2pqr
├── modified.pdb                  # phosphorylated — input to Stage 2
└── modification_manifest.json    # records which sites were patched
```

### Stage 2 — MD preparation (~1 minute)

1. **Topology** — runs `gmx pdb2gmx` with CHARMM36m to build the topology and GRO file; patches the topology file to include phospho-residue parameters
2. **Solvate** — runs `gmx editconf` (dodecahedron box, 1.2 nm padding) then `gmx solvate` (TIP3P water)
3. **Ionise** — runs `gmx grompp` + `gmx genion` to add NaCl at 150 mM and neutralise charge
4. **MDP files** — generates `minimization.mdp`, `nvt.mdp`, `npt.mdp`, `production.mdp` from the protocol YAML and simulation config

**Output files:**
```
output/stage2/
├── topol.top           # GROMACS topology
├── ions.gro            # solvated + ionised system (~10,000+ atoms)
├── minimization.mdp
├── nvt.mdp
├── npt.mdp
├── production.mdp
└── prep_report.json    # summary of system preparation
```

### Stage 3 — MD simulation (time varies by hardware — see the note at the top of this tutorial)

Runs four sequential GROMACS phases:

| Phase | Purpose | Default length |
|---|---|---|
| Minimization | Remove clashes, reach a local energy minimum | 50,000 steps (~steepest descent) |
| NVT equilibration | Stabilise temperature at 300 K | 500 ps |
| NPT equilibration | Stabilise pressure at 1 bar | 500 ps |
| Production | Collect statistics | Set by `production_time_ns` |

Each phase writes trajectory files to `output/stage3/<phase>/`.

**Key output:**
```
output/stage3/
└── production/
    ├── production.xtc   # trajectory (compressed, ~5–30 GB for 100 ns)
    ├── production.gro   # final frame
    ├── production.edr   # energy terms
    └── production.log   # mdrun log
```

### Stage 4 — Analysis (~1–5 minutes)

Loads the production trajectory with MDAnalysis and runs each plugin listed in `analysis.plugins`. Each plugin produces a CSV and a PNG, then everything is bundled into a single self-contained HTML report.

**Output:**
```
output/stage4/
├── report.html              # open this in a browser
├── rmsd.csv / rmsd.png
├── rmsf.csv / rmsf.png
├── radius_of_gyration.csv / radius_of_gyration.png
├── secondary_structure.csv / secondary_structure.png
├── hbond.csv / hbond.png
└── sasa.csv / sasa.png
```

---

## 6. Check progress and resume

If the pipeline is interrupted (power loss, timeout, etc.) you can check where it stopped:

```bash
phosp status output/
```

Example output:

```
          phosp Pipeline Status
┌────────────────────────────┬──────────────┬───────────────────────┬───────────────┐
│ Stage                      │ Status       │ Completed At          │ Key Artifacts │
├────────────────────────────┼──────────────┼───────────────────────┼───────────────┤
│ Stage 1 — Chemical Modif…  │ ✓ complete   │ 2026-06-30T09:12:44   │ modified.pdb  │
│ Stage 2 — MD Preparation   │ ✓ complete   │ 2026-06-30T09:13:21   │ topol.top     │
│ Stage 3 — MD Simulation    │ pending      │                       │               │
│ Stage 4 — Analysis         │ pending      │                       │               │
└────────────────────────────┴──────────────┴───────────────────────┴───────────────┘
```

Resume by running the same command again:

```bash
phosp run config.yaml
# Skipping stage1 (already complete)
# Skipping stage2 (already complete)
# Running stage3 ...
```

phosp picks up from stage 3 automatically. If a stage was partially completed when it was interrupted, the temp directory (`.stage3_tmp`) is cleaned up and the stage re-runs from scratch.

---

## 7. Open the report

```bash
# macOS
open output/stage4/report.html

# Linux
xdg-open output/stage4/report.html

# Or simply open the file in your browser
firefox output/stage4/report.html
```

The report contains embedded PNG plots for each analysis plugin. The figures are base64-encoded so the file is fully self-contained — you can email it or share it without any accompanying files.

If you want to regenerate the report without re-running the analysis (e.g. after editing the template):

```bash
phosp report output/
```

---

## 8. Interpreting the results

### RMSD

The RMSD plot shows Cα backbone root-mean-square deviation vs. time, measured relative to the first frame.

- A rising RMSD that levels off indicates the protein has reached equilibrium. The flat region is the part of the trajectory useful for analysis.
- An RMSD that keeps rising may indicate insufficient equilibration or a large conformational change.
- For ubiquitin, expect ~1–2 Å backbone RMSD at equilibrium.

### RMSF

Per-residue root-mean-square fluctuation shows which parts of the protein are flexible.

- Loop regions typically show higher RMSF than secondary structure elements (helices, sheets).
- The phosphorylated residue (TPO66 in this example) and its neighbours often show altered flexibility relative to the unphosphorylated protein.

### Radius of gyration

Rg measures the compactness of the protein.

- A stable Rg indicates the protein stays folded throughout the simulation.
- A decreasing Rg may indicate compaction; an increasing Rg may indicate unfolding.

### Secondary structure

Shows the fraction of residues in α-helix and β-sheet over time. A stable trajectory should show roughly constant secondary structure content.

---

## 9. Running a comparison

A common use case is comparing the phosphorylated protein against the wild-type. Use the `--reference` flag to run the same config's unmodified protein through the identical pipeline — no second config file needed:

```bash
# Run 1: phosphorylated (as configured in modification.sites)
phosp run config.yaml                # output → output/

# Run 2: unmodified — skips phosphorylation, same protocol otherwise
phosp run config.yaml --reference    # output → output_reference/
```

Both runs share the same config, so the protocol, box, water model, and analysis plugins stay identical — only the phospho-modification step differs. Each writes to its own output directory (`output/` vs `output_reference/`) and tracks its own checkpoint, so you can resume either independently.

Then load both trajectories in MDAnalysis or VMD for direct comparison. The CSV files from stage4 are easy to import into Python for custom plotting:

```python
import pandas as pd
import matplotlib.pyplot as plt

rmsd_p = pd.read_csv("output/stage4/rmsd.csv")
rmsd_wt = pd.read_csv("output_reference/stage4/rmsd.csv")

fig, ax = plt.subplots()
ax.plot(rmsd_p["time_ps"] / 1000, rmsd_p["rmsd_angstrom"], label="pThr66")
ax.plot(rmsd_wt["time_ps"] / 1000, rmsd_wt["rmsd_angstrom"], label="Wild-type")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("RMSD (Å)")
ax.legend()
plt.savefig("comparison.png", dpi=150)
```

---

## 10. Predicting phosphorylation sites

If you don't know which residues to phosphorylate, phosp can call NetPhos to predict candidate sites:

```bash
# NetPhos must be installed separately:
# https://services.healthtech.dtu.dk/software.php
phosp predict-sites config.yaml --threshold 0.7
```

Output:

```
  chain=A resid=10  resname=SER type=pSer score=0.941
  chain=A resid=20  resname=SER type=pSer score=0.812
  chain=A resid=66  resname=THR type=pThr score=0.889

Add selected entries to modification.sites in config.yaml
```

Copy the entries you want into `modification.sites` in your config.

---

## 11. Multi-site phosphorylation

Adding multiple phospho-sites is just a matter of listing more entries. All sites are applied in the order listed:

```yaml
modification:
  sites:
    - chain: A
      resid: 10
      resname: SER
      phospho_type: pSer
    - chain: A
      resid: 66
      resname: THR
      phospho_type: pThr
    - chain: B
      resid: 147
      resname: TYR
      phospho_type: pTyr
```

---

## 12. Longer simulations on a workstation

A production-quality simulation for a globular protein is typically 100–500 ns. For a small (~76-residue) protein on a capable GPU workstation, expect roughly:

| Duration | Approximate wall-clock time |
|---|---|
| 10 ns | 1–2 h (CPU) · 10–20 min (GPU) |
| 100 ns | 10–20 h (CPU) · 1–3 h (GPU) |
| 500 ns | 50–100 h (CPU) · 5–15 h (GPU) |

These numbers scale up substantially for larger proteins (roughly with total atom count, including solvent) — a ~500-residue protein can easily be 5-10x slower than this table on the same hardware. Confirm your `gmx mdrun` log actually reports GPU use with real performance numbers (`Performance: X ns/day`) before trusting a GPU estimate; see the note at the top of this tutorial about builds that silently fall back to CPU.

To use a GPU, GROMACS must be compiled with GPU support matching your hardware (CUDA for NVIDIA, ROCm/HIP for AMD — not every OpenCL build supports every GPU generation), or installed via conda with GPU enabled. Set `gpu_id: 0` (or another device index) in the config to pin to a specific GPU — phosp then also adds `-nb gpu -pme gpu -bonded gpu -update gpu` to `mdrun`, offloading nonbonded, PME, bonded, and update/constraint work, which is what actually saturates a datacenter GPU like an A100/H100/H200. Leaving `gpu_id: ~` still lets GROMACS auto-detect and use a GPU for nonbonded work, but the rest stays on CPU. On a multi-GPU node, each phosp run pins to one GPU (`-ntmpi 1`) — launch separate runs with different `gpu_id` values to use more than one card at once.

This tutorial covers running locally (`runner: local`). If you're on a SLURM/PBS cluster where the scheduler assigns the node and GPU automatically, the rules are different — see [HPC usage](../README.md#hpc-usage-slurm--pbs) in the README: leave `gpu_id: ~` there and set `hpc.gpus: 1` instead, since the offload flags key off "was a GPU requested" rather than a known device index.

---

## 13. Running on an HPC cluster

See [HPC usage in the README](../README.md#hpc-usage-slurm--pbs) and [HPC environments in the installation guide](installation.md#hpc-environments) for the full workflow.

The short version:

```bash
# 1. Run stages 1-2 on the login node
phosp run config.yaml --stages 1,2

# 2. Set runner: slurm in config, then run stage 3 to generate the job script
phosp run config.yaml --stages 3
sbatch output/stage3/run_slurm.sh

# 3. After the job finishes, run stage 4 on the login node
phosp run config.yaml --stages 4
```

---

## Appendix: quick test with a 1 ps run

To verify the full pipeline without waiting for equilibration:

```bash
# Create a minimal protocol
cat > ~/phosp_tutorial/quick.yaml << 'EOF'
box_padding_nm: 1.2
water_model: tip3p
salt_mM: 150
minimization:
  integrator: steep
  nsteps: 500
  emtol: 1000.0
  emstep: 0.01
nvt:
  dt: 0.002
  nsteps: 200
  tcoupl: V-rescale
  tc_grps: "Protein Non-Protein"
  tau_t: "0.1 0.1"
  ref_t: "300 300"
  define: -DPOSRES_HEAVY
npt:
  dt: 0.002
  nsteps: 200
  tcoupl: V-rescale
  tc_grps: "Protein Non-Protein"
  tau_t: "0.1 0.1"
  ref_t: "300 300"
  pcoupl: Parrinello-Rahman
  pcoupltype: isotropic
  tau_p: 2.0
  ref_p: 1.0
  compressibility: 4.5e-5
  define: -DPOSRES_BB
production:
  dt: 0.002
  tcoupl: V-rescale
  tc_grps: "Protein Non-Protein"
  tau_t: "0.1 0.1"
  ref_t: "300 300"
  pcoupl: Parrinello-Rahman
  pcoupltype: isotropic
  tau_p: 2.0
  ref_p: 1.0
  compressibility: 4.5e-5
  define: ""
EOF
```

In `config.yaml`:

```yaml
protocol: quick.yaml          # relative to where you run phosp
simulation:
  production_time_ns: 0.001   # 1 ps
```

The complete pipeline (all four stages) finishes in about 40 seconds.

---

## Summary of commands

```bash
phosp init config.yaml                     # generate starter config
phosp validate config.yaml                 # check config and dependencies
phosp run config.yaml                      # run full pipeline (or resume)
phosp run config.yaml --dry-run            # estimate disk, check tools, no execution
phosp run config.yaml --stages 1,2        # run specific stages
phosp run config.yaml --start-from stage3  # force-resume from a stage
phosp run config.yaml --reference          # run unmodified protein → output_reference/
phosp status output/                       # show which stages are complete, with duration
phosp report output/                       # regenerate HTML report
phosp predict-sites config.yaml            # predict phospho sites (needs NetPhos)
phosp clean output/                        # remove an output dir + checkpoint (with confirmation)
```
