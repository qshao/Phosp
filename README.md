# phosp

Automated pipeline for phosphorylation + GROMACS MD simulation. Given a protein structure and one or more phosphorylation sites, phosp handles everything from chemical modification through production MD to analysis.

```
Stage 1  Chemical modification   THR → TPO (pThr), SER → SEP (pSer), TYR → PTR (pTyr)
Stage 2  MD preparation          pdb2pqr protonation → pdb2gmx topology → solvation → ions
Stage 3  MD simulation           Minimization → NVT → NPT → production
Stage 4  Analysis                RMSD, RMSF, Rg, secondary structure, H-bonds, SASA + HTML report
```

## Prerequisites

| Dependency | Version | Install |
|---|---|---|
| Python | ≥ 3.10 | — |
| GROMACS | ≥ 2022 | `conda install -c conda-forge gromacs` |
| pdb2pqr | ≥ 3.0 | `pip install pdb2pqr` |
| CHARMM36m FF | jul2022 | see below |

### Installing the CHARMM36m force field

GROMACS does not ship CHARMM36m. Download and install it manually:

```bash
# 1. Download from MacKerell lab
curl -O https://mackerell.umaryland.edu/download.php?filename=CHARMM_ff_params_files/charmm36-jul2022.ff.tgz

# 2. Extract into GROMACS topology directory
GMXTOP=$(gmx --version 2>&1 | awk '/Data prefix/{print $3}')/share/gromacs/top
tar -xzf charmm36-jul2022.ff.tgz -C "$GMXTOP"

# 3. Create the charmm36m symlink phosp expects
ln -s "$GMXTOP/charmm36-jul2022.ff" "$GMXTOP/charmm36m-jul2022.ff"

# 4. Register phospho-residues as protein type
printf "SEP\tProtein\nTPO\tProtein\nPTR\tProtein\n" >> "$GMXTOP/residuetypes.dat"

# 5. Fix a naming collision in the ether terminal database
sed -i 's/^\[ MET1 \]/[ EMETH1 ]/' "$GMXTOP/charmm36-jul2022.ff/ethers.n.tdb"
```

> **Note:** `charmm36m-jul2022.ff.tgz` (the `m` variant) is not publicly available from MacKerell's website — only `charmm36-jul2022.ff.tgz` is. Steps 3–5 make it work identically for protein simulations.

## Installation

```bash
git clone https://github.com/qshao/Phosp.git
cd Phosp
pip install -e .
```

## Quick start

```bash
# 1. Generate a config file
phosp init my_run/config.yaml

# 2. Edit the config (set input path, phospho sites, simulation length)

# 3. Validate before running
phosp validate my_run/config.yaml

# 4. Run the full pipeline
phosp run my_run/config.yaml

# 5. Check status / re-open the report
phosp status my_run/config.yaml
open my_run/output/stage4/report.html
```

The pipeline is **checkpoint-aware** — if it stops at any stage, re-running the same command resumes from where it left off. To re-run from a specific stage:

```bash
phosp run my_run/config.yaml --start-from stage3
```

## Examples

Ready-to-run configs are in `examples/`:

| Example | Description |
|---|---|
| `examples/ubiquitin_pThr/` | pThr66 on ubiquitin — quickest end-to-end test |
| `examples/multisite_pSer/` | Two pSer sites, full analysis suite |
| `examples/uniprot_input/` | Fetch structure from AlphaFold/RCSB by UniProt ID |

```bash
# Run the bundled ubiquitin example (uses tests/fixtures/ubiquitin.pdb)
phosp run examples/ubiquitin_pThr/run.yaml
```

## Configuration reference

```yaml
input:
  source: pdb            # "pdb" (local file) or "uniprot" (fetch from AlphaFold/RCSB)
  path: protein.pdb      # required when source=pdb
  uniprot_id: P62988     # required when source=uniprot
  ph: 7.4                # pH for protonation assignment (pdb2pqr)

modification:
  sites:
    - chain: A           # PDB chain ID
      resid: 66          # residue number
      resname: THR       # SER, THR, or TYR (must match the PDB)
      phospho_type: pThr # pSer, pThr, or pTyr

forcefield: charmm36m    # "charmm36m" or "amber_ff14sb"

protocol: globular_protein   # built-in preset name, or path to a custom YAML
# Built-in presets: globular_protein | membrane_protein | phosphopeptide

simulation:
  production_time_ns: 100.0
  output_freq_ps: 10.0
  water_model: tip3p          # tip3p | spce
  box_type: dodecahedron      # dodecahedron | cubic
  salt_concentration_mM: 150.0
  hpc:
    enabled: false
    scheduler: slurm           # slurm | pbs
    ntasks: 8
    gpus: 1
    walltime: "24:00:00"
    partition: gpu
    auto_submit: false

analysis:
  plugins:
    - rmsd
    - rmsf
    - radius_of_gyration
    - secondary_structure
    - hbond
    - contacts
    - sasa
  rmsd:
    selection: backbone
  rmsf:
    selection: name CA
  sasa:
    residues: []               # empty = all residues
```

### Custom protocol files

Copy and edit a built-in preset (found in `phosp/protocols/`):

```bash
cp phosp/protocols/globular_protein.yaml my_project/fast_protocol.yaml
# Edit nsteps values, then reference it in your config:
# protocol: my_project/fast_protocol.yaml
```

## CLI reference

```
phosp run <config>               Run the full pipeline (or resume from checkpoint)
  --start-from stage2            Resume from a specific stage
  --stages 1,2                   Run only the listed stages
  --dry-run                      Validate config + check tools, don't run
  --log-level INFO               DEBUG | INFO | WARNING | ERROR
  --log-file run.log             Write logs to file

phosp validate <config>          Check config syntax and tool availability
phosp predict-sites <config>     Predict phosphorylation sites (requires NetPhos)
phosp status <config>            Show completed stages and checkpoint state
phosp report <output-dir>        Regenerate the HTML report from existing results
phosp init [path]                Write a template config file
```

## Analysis plugins

| Plugin | Output |
|---|---|
| `rmsd` | RMSD vs. time CSV + PNG |
| `rmsf` | Per-residue RMSF CSV + PNG |
| `radius_of_gyration` | Rg vs. time CSV + PNG |
| `secondary_structure` | Helix/sheet fraction vs. time |
| `hbond` | Hydrogen bond count vs. time |
| `contacts` | Native contact fraction Q vs. time |
| `sasa` | Solvent-accessible surface area vs. time |

All results are collected in an HTML report at `output/stage4/report.html`.

## HPC usage

Set `hpc.enabled: true` in the config. Stage 2 generates a job script (`run_slurm.sh` or `run_pbs.sh`) in the output directory. With `auto_submit: true` the script is submitted automatically; otherwise submit it manually:

```bash
sbatch output/stage3/run_slurm.sh
```

## Running tests

```bash
pip install -e ".[dev]"
pytest
```
