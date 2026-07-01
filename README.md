# phosp

Automated pipeline for protein phosphorylation + GROMACS molecular dynamics simulation.
Given a protein structure and one or more phosphorylation sites, phosp handles everything from chemical modification through production MD to a self-contained HTML analysis report — in a single command.

```
Stage 1  Chemical modification    THR → TPO (pThr),  SER → SEP (pSer),  TYR → PTR (pTyr)
Stage 2  MD preparation           pdb2pqr protonation → topology (pdb2gmx) → solvation → ions
Stage 3  MD simulation            Minimization → NVT equilibration → NPT equilibration → Production
Stage 4  Analysis                 RMSD · RMSF · Rg · 2° structure · H-bonds · SASA · PCA · DCCM + HTML report
```

The pipeline is **checkpoint-aware**: if it stops at any stage, re-running the same command resumes from where it left off.

---

## Table of contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [CHARMM36m force field setup](#charmm36m-force-field-setup)
4. [Quick start](#quick-start)
5. [Examples](#examples)
6. [Configuration reference](#configuration-reference)
7. [CLI reference](#cli-reference)
8. [Analysis plugins](#analysis-plugins)
9. [Simulation protocols](#simulation-protocols)
10. [HPC usage (SLURM / PBS)](#hpc-usage-slurm--pbs)
11. [Troubleshooting](#troubleshooting)

For a step-by-step walkthrough see [docs/tutorial.md](docs/tutorial.md).
For detailed environment and dependency setup see [docs/installation.md](docs/installation.md).

---

## Requirements

| Dependency  | Minimum version | Notes |
|---|---|---|
| Python | 3.10 | 3.11 or 3.12 recommended |
| GROMACS | 2022 | 2024 or 2026 tested |
| pdb2pqr | 3.0 | `pip install pdb2pqr` |
| CHARMM36m | jul2022 | see [setup below](#charmm36m-force-field-setup) |

Python package dependencies (installed automatically):

| Package | Purpose |
|---|---|
| MDAnalysis | trajectory analysis |
| Biopython | PDB parsing and manipulation |
| Pydantic v2 | config validation |
| Typer | CLI |
| Jinja2 | MDP and HPC script templating |
| matplotlib, pandas, numpy | plotting and data |
| freesasa | solvent-accessible surface area |
| PyYAML | config parsing |

---

## Installation

### 1. Create a virtual environment

**With conda (recommended)** — installs GROMACS in the same environment:

```bash
conda create -n phosp python=3.12 -y
conda activate phosp
conda install -c conda-forge gromacs -y
```

**With venv** — if you have GROMACS installed separately:

```bash
python3.12 -m venv ~/envs/phosp
source ~/envs/phosp/bin/activate       # Linux / macOS
# .\envs\phosp\Scripts\activate        # Windows
```

### 2. Clone and install phosp

```bash
git clone https://github.com/qshao/Phosp.git
cd Phosp
pip install -e .                        # installs all Python dependencies
```

For development (adds pytest and ruff):

```bash
pip install -e ".[dev]"
```

### 3. Install pdb2pqr

```bash
pip install pdb2pqr
```

Verify: `pdb2pqr --version` should print `3.x.x`.

### 4. Verify the installation

```bash
phosp --version
gmx --version | head -3
pdb2pqr --version
```

See [docs/installation.md](docs/installation.md) for platform-specific notes and troubleshooting.

---

## CHARMM36m force field setup

GROMACS does not ship CHARMM36m. Run these five commands once after installing GROMACS:

```bash
# 1. Download from MacKerell lab
curl -O "https://mackerell.umaryland.edu/download.php?filename=CHARMM_ff_params_files/charmm36-jul2022.ff.tgz" \
     -o charmm36-jul2022.ff.tgz

# 2. Find the GROMACS topology directory
GMXTOP=$(gmx --version 2>&1 | awk '/Data prefix/{print $3}')/share/gromacs/top

# 3. Extract the force field
tar -xzf charmm36-jul2022.ff.tgz -C "$GMXTOP"

# 4. Create the symlink phosp expects
ln -s "$GMXTOP/charmm36-jul2022.ff" "$GMXTOP/charmm36m-jul2022.ff"

# 5. Register phospho-residue types so pdb2gmx recognises SEP / TPO / PTR
printf "SEP\tProtein\nTPO\tProtein\nPTR\tProtein\n" >> "$GMXTOP/residuetypes.dat"

# 6. Fix a naming collision in the ether terminal database
sed -i 's/^\[ MET1 \]/[ EMETH1 ]/' "$GMXTOP/charmm36-jul2022.ff/ethers.n.tdb"
```

> **Note:** Only `charmm36-jul2022.ff.tgz` (without the `m`) is publicly distributed from MacKerell's website. Steps 4–6 make it fully compatible with phosp's CHARMM36m support for protein simulations.

Verify with:

```bash
phosp validate examples/ubiquitin_pThr/run.yaml
```

---

## Quick start

```bash
# 1. Generate a starter config
phosp init my_run/config.yaml

# 2. Edit the config: set input.path, modification.sites, production_time_ns
#    (see Configuration reference below)

# 3. Validate before running
phosp validate my_run/config.yaml

# 4. Run the full pipeline
phosp run my_run/config.yaml

# 5. View the report
open my_run/output/stage4/report.html   # macOS
xdg-open my_run/output/stage4/report.html   # Linux
```

The pipeline writes all output to `<config-dir>/output/`:

```
output/
├── checkpoint.json       # resume state
├── stage1/
│   ├── modified.pdb      # phosphorylated structure
│   └── modification_manifest.json
├── stage2/
│   ├── topol.top         # GROMACS topology
│   ├── ions.gro          # solvated + ionized system
│   └── *.mdp             # MD parameter files
├── stage3/
│   └── production/
│       ├── production.xtc   # trajectory
│       └── production.gro   # final frame
└── stage4/
    ├── report.html       # interactive analysis report
    ├── rmsd.csv / rmsd.png
    ├── rmsf.csv / rmsf.png
    └── ...
```

---

## Examples

Three ready-to-run configs are in `examples/`:

| Example | What it shows |
|---|---|
| `examples/ubiquitin_pThr/` | pThr66 on ubiquitin — fastest end-to-end test |
| `examples/multisite_pSer/` | Two pSer sites with a full analysis suite |
| `examples/uniprot_input/` | Fetch structure from AlphaFold / RCSB by UniProt ID |

```bash
# Quickest end-to-end test (uses bundled ubiquitin PDB, ~40 s with the quick protocol)
phosp run examples/ubiquitin_pThr/run.yaml
```

---

## Configuration reference

A config file is a YAML document. Generate a template with `phosp init`.

### `input` block

```yaml
input:
  source: pdb             # "pdb" — local file; "uniprot" — fetch from AlphaFold/RCSB
  path: protein.pdb       # required when source=pdb
  # uniprot_id: P62988    # required when source=uniprot
  ph: 7.4                 # pH for protonation state assignment (pdb2pqr / PROPKA)
```

When `source: uniprot`, phosp queries the AlphaFold prediction API for the current model version (AlphaFold periodically re-predicts entries under a new version, so this is looked up dynamically rather than hardcoded). If AlphaFold has no entry, it queries RCSB Search and downloads the top experimental structure instead.

### `modification` block

```yaml
modification:
  sites:
    - chain: A            # PDB chain ID (case-sensitive)
      resid: 66           # residue sequence number as it appears in the PDB
      resname: THR        # must be SER, THR, or TYR — must match the PDB
      phospho_type: pThr  # pSer (→ SEP), pThr (→ TPO), pTyr (→ PTR)
```

Multiple sites can be listed. Pydantic validates that `resname` and `phospho_type` are consistent (e.g. `SER` + `pSer`).

### `forcefield`

```yaml
forcefield: charmm36m     # only supported option; amber_ff14sb is not yet available
```

### `gromacs` block

```yaml
gromacs:
  binary: gmx              # binary name or full path, e.g. "gmx_mpi" or "/opt/gromacs/bin/gmx"
  pdb2pqr: pdb2pqr         # binary name or full path; useful for non-PATH installations
  timeout_minutes: ~       # hard cap per GROMACS/pdb2pqr/NetPhos/gmx_MMPBSA subprocess call;
                           # ~ = no limit (default); e.g. 120 for a 2-hour cap
```

### `protocol`

```yaml
protocol: globular_protein   # built-in preset name, or path to a custom YAML file
```

Built-in presets in `phosp/protocols/`:

| Preset | Suitable for |
|---|---|
| `globular_protein` | Standard soluble proteins |
| `membrane_protein` | Membrane-embedded proteins (rectangular box, larger padding) |
| `phosphopeptide` | Short peptides |

To use a custom protocol, point to your YAML file:

```yaml
protocol: my_project/fast_protocol.yaml
```

Copy a built-in preset and edit the `nsteps` values:

```bash
cp phosp/protocols/globular_protein.yaml my_project/fast_protocol.yaml
```

### `simulation` block

```yaml
simulation:
  production_time_ns: 100.0       # total production run length
  output_freq_ps: 10.0            # trajectory write frequency
  water_model: tip3p              # "tip3p" or "spce"
  box_type: dodecahedron          # "dodecahedron" or "cubic"
  salt_concentration_mM: 150.0    # NaCl concentration
  gpu_id: ~                       # GPU index (0, 1, …); ~ = GROMACS auto-selects
  runner: local                   # "local", "slurm", or "pbs" — see HPC section

  hpc:                            # only used when runner is slurm or pbs
    ntasks: 8                     # OpenMP threads per rank (maps to --cpus-per-task in SLURM)
    gpus: 1
    walltime: "24:00:00"
    partition: gpu                # cluster queue / partition name
    auto_submit: false            # true = sbatch/qsub automatically; false = write script only
    # gromacs_module: gromacs/2026.0-cuda   # environment module to load; omit if not needed
    # extra_directives:                      # any additional scheduler options
    #   - "--account=myproject"
    #   - "--qos=high"
    #   - "--constraint=a100"
    #   - "--mem=128G"
```

### `analysis` block

```yaml
analysis:
  plugins:
    - rmsd
    - rmsf
    - radius_of_gyration
    - secondary_structure
    - hbond
    - contacts
    - sasa
    - pca
    - dccm
    - salt_bridges
    # - mmpbsa          # requires gmx_MMPBSA (pip install gmx_MMPBSA)

  rmsd:
    selection: backbone             # MDAnalysis atom selection string
  rmsf:
    selection: name CA
  sasa:
    residues: []                    # [] = all residues; [10, 47] = specific residues
  contacts:
    selection: name CA
    cutoff_angstrom: 8.0
  salt_bridges:
    cutoff_angstrom: 4.0
  mmpbsa:
    method: pbsa                    # "pbsa" or "gbsa"
    temperature: 300
```

---

## CLI reference

```
phosp run <config>
  Run the full pipeline. Resumes from checkpoint automatically.
  --start-from stage2     Resume from a specific stage (stage1–stage4)
  --stages 1,2            Run only the listed stage numbers
  --dry-run               Validate config + check tools, print disk estimate, exit
  --log-level INFO        DEBUG | INFO | WARNING | ERROR (default: INFO)
  --log-file run.log      Write structured log to this file in addition to stdout
  --reference              Run the unmodified protein instead of applying
                           modification.sites — skips phosphorylation, uses
                           the same protocol. Output goes to
                           <config-dir>/output_reference/ so it coexists
                           with a normal run for direct comparison.

phosp validate <config>
  Parse config, check GROMACS binary, pdb2pqr, and CHARMM36m installation.
  Exits 0 on success, 1 on failure.

phosp predict-sites <config>
  Run NetPhos to predict phosphorylatable S/T/Y residues.
  Requires NetPhos installed and on PATH (https://services.healthtech.dtu.dk/).
  --threshold 0.5         Minimum confidence score (0–1)

phosp status <output-dir>
  Show a table of completed stages, timestamps, key artifacts, and
  per-stage duration. output-dir is usually <config-dir>/output/ (or
  output_reference/ for a --reference run).

phosp report <output-dir>
  Regenerate the HTML report from existing stage4 results without re-running analysis.

phosp init [path]
  Write a starter config YAML to the given path (default: phosp_config.yaml).

phosp clean <output-dir>
  Remove a pipeline output directory (and its checkpoint) after a
  confirmation prompt. Use this to force a clean re-run instead of
  resuming from checkpoint.

phosp --version
  Print the installed version and exit.
```

### Resume and partial runs

```bash
# Resume a run that stopped mid-way through stage3
phosp run my_run/config.yaml

# Force re-run from stage3 even if it completed previously
phosp run my_run/config.yaml --start-from stage3

# Run only stages 1 and 2 (e.g. to prepare a system before editing the protocol)
phosp run my_run/config.yaml --stages 1,2
```

### Reference (wild-type) comparison runs

```bash
# Runs the same config's unmodified protein through the identical pipeline,
# skipping the phospho-modification step. Writes to output_reference/, so it
# coexists with a normal --reference-less run of the same config.
phosp run my_run/config.yaml --reference
```

Compare the two runs' `stage4/*.csv` outputs directly — see [Running a comparison](docs/tutorial.md#9-running-a-comparison) in the tutorial for an example.

---

## Analysis plugins

All plugins produce a CSV file and a PNG plot, collected into the HTML report.

| Plugin | Config key | Output | Notes |
|---|---|---|---|
| `rmsd` | `rmsd.selection` | RMSD vs. time | default selection: `backbone` |
| `rmsf` | `rmsf.selection` | Per-residue RMSF | default: `name CA` |
| `radius_of_gyration` | — | Rg vs. time | |
| `secondary_structure` | — | Helix / sheet fraction vs. time | |
| `hbond` | — | H-bond count vs. time | |
| `contacts` | `contacts.selection`, `contacts.cutoff_angstrom` | Cα contact count | default cutoff: 8 Å |
| `sasa` | `sasa.residues` | SASA vs. time | per-residue or total |
| `pca` | `pca.selection` | PC1 vs. PC2 scatter | default: `name CA` |
| `dccm` | `dccm.selection` | Dynamic cross-correlation matrix | |
| `salt_bridges` | `salt_bridges.cutoff_angstrom` | Salt-bridge count vs. time | default cutoff: 4 Å |
| `mmpbsa` | `mmpbsa.method`, `mmpbsa.temperature` | Per-residue ΔG | requires `gmx_MMPBSA` |

Any plugin that fails is skipped and reported in the HTML summary; other plugins continue.

---

## Simulation protocols

A protocol file sets the MD parameters for each phase. The `simulation` config block controls the system-level settings (box, water, salt, run length); the protocol sets the integrator details.

```yaml
# example: phosp/protocols/globular_protein.yaml (excerpt)
minimization:
  integrator: steep
  nsteps: 50000
  emtol: 1000.0      # kJ/mol/nm — convergence threshold

nvt:
  dt: 0.002          # ps (2 fs time step)
  nsteps: 250000     # 500 ps equilibration
  tcoupl: V-rescale
  ref_t: "300 300"   # K — protein + solvent coupling groups

npt:
  dt: 0.002
  nsteps: 250000     # 500 ps equilibration
  pcoupl: Parrinello-Rahman
  ref_p: 1.0         # bar

production:
  dt: 0.002
  # nsteps is derived from simulation.production_time_ns
```

For a quick test, copy and reduce `nsteps`:

```bash
cp phosp/protocols/globular_protein.yaml my_run/quick.yaml
# edit quick.yaml: set minimization.nsteps: 500, nvt.nsteps: 200, npt.nsteps: 200
```

Then in the config:

```yaml
protocol: my_run/quick.yaml
simulation:
  production_time_ns: 0.001   # 1 ps — just to confirm everything runs
```

---

## HPC usage (SLURM / PBS)

Set `runner: slurm` (or `runner: pbs`) in the config. Stage 3 generates a job script instead of running GROMACS directly.

```yaml
gromacs:
  binary: gmx_mpi            # MPI-enabled binary common on HPC clusters

simulation:
  runner: slurm
  hpc:
    ntasks: 32               # OpenMP threads per rank (SLURM: --cpus-per-task)
    gpus: 1
    walltime: "48:00:00"
    partition: gpu
    auto_submit: false        # set true to submit automatically
    gromacs_module: gromacs/2026.0-cuda   # adjust to match your cluster's module name
    extra_directives:         # any additional SLURM options your cluster requires
      - "--account=myproject"
      - "--qos=high"
      - "--constraint=a100"
      - "--mem=128G"
```

### Workflow

**Step 1** — Run stages 1 and 2 on the login node:

```bash
phosp run my_run/config.yaml --stages 1,2
```

**Step 2** — Run stage 3, which generates (and optionally submits) the HPC job:

```bash
phosp run my_run/config.yaml --stages 3
# → writes my_run/output/stage3/run_slurm.sh  (or run_pbs.sh)
# → writes my_run/output/stage3/pending_job.json

sbatch my_run/output/stage3/run_slurm.sh     # if auto_submit: false
```

**Step 3** — After the job finishes, run stage 4:

```bash
phosp run my_run/config.yaml --stages 4
```

If you try to run stage 4 while the job is still running, phosp detects `pending_job.json` and prints a clear error with the job ID and the command to check status (`squeue` / `qstat`).

### Checking job status

```bash
squeue -j <job_id>         # SLURM
qstat <job_id>             # PBS
```

---

## Troubleshooting

**`CHARMM36m force field not found`**
Run the six-step [setup](#charmm36m-force-field-setup) above. Verify with `phosp validate <config>`.

**`Phospho-residue types [SEP / TPO / PTR] missing from residuetypes.dat`**
Run step 5 of the force field setup:
```bash
GMXTOP=$(gmx --version 2>&1 | awk '/Data prefix/{print $3}')/share/gromacs/top
printf "SEP\tProtein\nTPO\tProtein\nPTR\tProtein\n" >> "$GMXTOP/residuetypes.dat"
```

**`Config validation failed: modification.sites.0: …`**
`resname` must match `phospho_type` exactly: `SER`/`pSer`, `THR`/`pThr`, `TYR`/`pTyr`.

**`GROMACS binary 'gmx' not found`**
Either install GROMACS or set `gromacs.binary` to the correct path/name in the config.

**Stage stops with `pdb2pqr not found in PATH`**
```bash
pip install pdb2pqr
```

**Pipeline stopped mid-stage and left a `.stage3_tmp` directory**
This is an orphaned temp dir. phosp cleans it automatically on the next run. If you want to clean manually:
```bash
rm -rf my_run/output/.stage3_tmp
```

**Want to force a completely clean re-run instead of resuming from checkpoint**
```bash
phosp clean my_run/output/
phosp run my_run/config.yaml
```

**`production.xtc not found — HPC job is still running`**
The stage 3 SLURM/PBS job hasn't completed yet. Check with `squeue` or `qstat`, then re-run `phosp run` after the job finishes.

**`AMBER ff14SB is not yet fully supported`**
Use `forcefield: charmm36m` (the only currently supported force field).

**Memory error in MDAnalysis during analysis**
Reduce the number of analysis plugins or run them one at a time using `analysis.plugins: [rmsd]` etc.

---

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

149 tests, ~5 s on a laptop. No GROMACS or pdb2pqr required for the test suite (all external calls are mocked).

---

## Project structure

```
phosp/
├── cli.py                  # Typer CLI entry point
├── config.py               # Pydantic config models
├── pipeline.py             # Orchestrator (preflight, stage dispatch, checkpoint)
├── stages/                 # Stage 1–4 implementations
├── engines/gromacs.py      # GROMACS subprocess wrapper
├── runners/                # local.py, slurm.py, pbs.py — simulation backends
├── forcefields/            # CHARMM36m parameter handling
├── modification/           # pSer / pThr / pTyr chemical modification
├── plugins/analysis/       # All analysis plugins
├── protocols/              # Built-in MD protocol YAML files
├── templates/              # Jinja2 templates for MDP, HPC scripts, HTML report
├── prediction/netphos.py   # NetPhos phospho-site prediction wrapper
└── utils/                  # checkpoint.py, structure.py
examples/                   # Ready-to-run config files
docs/                       # Tutorial and installation guide
tests/                      # Unit and integration tests
```
