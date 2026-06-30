# Phosp: Automated Phosphorylation + MD Simulation Framework — Design Spec

**Date:** 2026-06-30  
**Status:** Approved  

---

## 1. Overview

`phosp` is a Python framework that automates four sequential stages for studying protein phosphorylation via MD simulation:

1. **Stage 1 — Chemical Modification:** acquire a protein structure, apply phospho patches to target residues
2. **Stage 2 — MD Preparation:** build the simulation system with correct force field parameters
3. **Stage 3 — MD Simulation:** run minimization, equilibration, and production MD
4. **Stage 4 — Analysis:** compute and report a configurable set of structural/dynamic properties

A single YAML config file drives the entire pipeline. A `Pipeline` orchestrator executes stages in order, writes checkpoints after each, and can resume from any stage without restarting from scratch.

---

## 2. Scope & Constraints

- **Initial engine:** GROMACS (abstracted behind `MDEngine` interface for future engines)
- **Initial force fields:** CHARMM36m and AMBER ff14SB + GAFF2 (abstracted behind `ForceField` interface)
- **Supported phosphorylation types:** pSer, pThr, pTyr
- **Input formats:** PDB file or UniProt ID (AlphaFold DB → RCSB fallback)
- **Python version:** ≥ 3.10
- **Key dependencies:** MDAnalysis, BioPython, Pydantic v2, Typer, Jinja2, Matplotlib, PDB2PQR (external), GROMACS (external), gmx_MMPBSA (optional external, required for `mmpbsa` plugin), NetPhos (optional external)

---

## 3. Package Structure

```
phosp/
├── cli.py                        # Typer CLI entry point
├── pipeline.py                   # Pipeline orchestrator + checkpointing
├── config.py                     # Pydantic config schema + YAML loader
├── stages/
│   ├── base.py                   # Abstract Stage base class
│   ├── stage1_modify.py          # Chemical modification
│   ├── stage2_prepare.py         # MD file preparation
│   ├── stage3_simulate.py        # MD simulation runner
│   └── stage4_analyze.py         # Analysis dispatcher
├── engines/
│   ├── base.py                   # Abstract MDEngine interface
│   └── gromacs.py                # GROMACS subprocess wrapper
├── forcefields/
│   ├── base.py                   # Abstract ForceField interface
│   ├── charmm36m.py              # CHARMM36m + phospho patches
│   └── amber_ff14sb.py           # AMBER ff14SB + GAFF2 + Homeyer frcmod
├── modification/
│   ├── base.py                   # Abstract Modifier base class
│   ├── pser.py                   # pSer patch logic
│   ├── pthr.py                   # pThr patch logic
│   └── ptyr.py                   # pTyr patch logic
├── prediction/
│   └── netphos.py                # NetPhos CLI/API wrapper
├── protocols/
│   ├── globular_protein.yaml     # Default physiological preset
│   ├── phosphopeptide.yaml       # Short peptide preset
│   └── membrane_protein.yaml     # Membrane system preset
├── plugins/
│   └── analysis/                 # Auto-discovered analysis plugins
│       ├── base.py               # AnalysisPlugin ABC
│       ├── rmsd.py
│       ├── rmsf.py
│       ├── radius_of_gyration.py
│       ├── secondary_structure.py
│       ├── hbond.py
│       ├── salt_bridges.py
│       ├── contacts.py
│       ├── mmpbsa.py
│       ├── sasa.py
│       ├── pca.py
│       └── dccm.py
├── templates/
│   └── report.html.j2            # Jinja2 HTML report template
└── utils/
    ├── structure.py              # PDB fetching, cleaning, protonation
    └── checkpoint.py            # Checkpoint save/load (JSON)
```

---

## 4. Core Abstractions

### 4.1 Stage Base Class

```python
class Stage(ABC):
    def __init__(self, config: PhospConfig, engine: MDEngine, forcefield: ForceField): ...
    def validate_inputs(self) -> None: ...       # raises StageInputError if invalid
    def run(self) -> StageResult: ...            # main logic
```

The `Pipeline` orchestrator calls `validate_inputs()` then `run()` for each stage. After each successful `run()`, it writes a checkpoint entry so the pipeline can resume mid-run.

### 4.2 MDEngine Interface

```python
class MDEngine(ABC):
    def prepare_topology(self, pdb, forcefield) -> Path: ...
    def solvate(self, topology, box_type, water_model) -> Path: ...
    def add_ions(self, topology, concentration_mM, neutralize) -> Path: ...
    def generate_mdp(self, phase, protocol) -> Path: ...
    def run_phase(self, phase, tpr) -> SimulationResult: ...
    def generate_hpc_script(self, scheduler, resources) -> Path: ...
```

`GROMACSEngine` wraps each method as a subprocess call (`gmx <subcommand>`), parses stdout/stderr, and raises `SimulationError` on non-zero exit.

### 4.3 ForceField Interface

```python
class ForceField(ABC):
    name: str
    def get_residue_params(self, resname: str) -> Path: ...      # standard residues
    def get_phospho_params(self, phospho_type: str) -> Path: ... # pSer/pThr/pTyr patches
    def patch_topology(self, topology: Path, sites: list[PhosphoSite]) -> Path: ...
```

### 4.4 AnalysisPlugin Interface

```python
class AnalysisPlugin(ABC):
    name: str                                   # matches config key
    def run(self, universe: mda.Universe, config: dict) -> pd.DataFrame: ...
    def plot(self, result: pd.DataFrame) -> Figure: ...
```

Plugins are auto-discovered at import time via `importlib` + `pkgutil` scanning `plugins/analysis/`. Users add custom plugins by placing a file in that directory — no core code changes required.

---

## 5. Stage Designs

### 5.1 Stage 1 — Chemical Modification

**Inputs:** PDB path or UniProt ID; site list (config or prediction mode); target pH.

**Steps:**
1. **Structure acquisition:** If UniProt ID, query AlphaFold DB API; fall back to RCSB PDB search.
2. **Cleaning:** Strip non-essential HETATMs (configurable keep-list). Warn on missing residues or chain breaks.
3. **Protonation:** Call PDB2PQR at target pH (default 7.4) to assign protonation states and add hydrogens.
4. **Site selection:** parse `modification.sites` from YAML into validated `PhosphoSite` objects. This is the only path used during `phosp run`. NetPhos prediction is a separate, user-invoked step (see CLI Section 7) that writes a suggested site list for the user to review and paste into their config — it never runs automatically as part of the pipeline.
5. **Patch application:** Each `pSer/pThr/pTyr` modifier reads force-field-specific patch templates bundled with `phosp` and applies coordinate + atom-name transformations.

**Outputs:** `output/stage1/modified.pdb`, `modification_manifest.json`, `protonation_report.txt`

---

### 5.2 Stage 2 — MD File Preparation

**Inputs:** `modified.pdb`, `modification_manifest.json`, force field selection, protocol preset.

**Steps:**
1. **Parameter assignment:** `ForceField.patch_topology()` merges standard and phospho-residue parameters into a single topology.
2. **System building (GROMACS):**
   - `gmx pdb2gmx` → initial topology + processed structure
   - `gmx editconf` → periodic box (dodecahedron default; cubic optional via config)
   - `gmx solvate` → TIP3P water (SPC/E selectable)
   - `gmx genion` → 150 mM NaCl + charge neutralization
3. **MDP generation:** `Protocol` class renders templated `.mdp` files for each phase (minimization, NVT, NPT, production) from the named preset YAML. Preset values are fully overridable in the master config.

**Protocol presets (bundled):**

| Preset | Box | Water | Salt | Production default |
|---|---|---|---|---|
| `globular_protein` | Dodecahedron, 1.2 nm | TIP3P | 150 mM NaCl | 100 ns |
| `phosphopeptide` | Cubic, 1.0 nm | TIP3P | 150 mM NaCl | 50 ns |
| `membrane_protein` | Rectangular | TIP3P | 150 mM NaCl | 200 ns (basic preset; full membrane builder out of scope for v1) |

**Outputs:** Complete GROMACS run directory: `.top`, `.itp`, `.gro`, all `.mdp` files, `prep_report.json`

---

### 5.3 Stage 3 — MD Simulation

**Inputs:** Prepared GROMACS run directory from Stage 2.

**Simulation phases (sequential):**

| Phase | Method | Duration |
|---|---|---|
| Energy minimization | Steepest descent, max 50k steps | Until Fmax < 1000 kJ/mol/nm |
| NVT equilibration | V-rescale thermostat, 300 K | 50 ns (heavy atom restraints) |
| NPT equilibration | Parrinello-Rahman barostat, 1 bar | 50 ns (backbone restraints) |
| Production | NPT, no restraints | Config-specified (default 100 ns) |

Each phase: `gmx grompp` to generate `.tpr`, then `gmx mdrun`. On non-zero exit, `SimulationError` is raised with the last 50 lines of the GROMACS log.

**HPC mode:** When `hpc.enabled: true`, renders a SLURM or PBS job script (scheduler configurable). `hpc.auto_submit: true` submits via `sbatch`/`qsub`; otherwise script is written for manual submission.

**Progress monitoring:** Pipeline tails the GROMACS `.log` file and emits structured progress lines (step / total steps, estimated time remaining).

**Checkpointing:** A checkpoint entry is written after each phase. Interrupted runs resume from the last completed phase.

**Output directory layout:**
```
output/stage3/
├── minimization/   em.gro, em.log, em.edr
├── nvt/            nvt.gro, nvt.log, nvt.edr, nvt.cpt
├── npt/            npt.gro, npt.log, npt.edr, npt.cpt
└── production/     prod.xtc, prod.gro, prod.log, prod.edr, prod.cpt
```

---

### 5.4 Stage 4 — Analysis

**Inputs:** `prod.xtc`, `prod.tpr`, `analysis` config block.

**Plugin discovery:** `Stage4Analyze` scans `plugins/analysis/` at import time and builds a registry of all `AnalysisPlugin` subclasses keyed by `name`. Only plugins listed in `analysis.plugins` are executed.

**Built-in plugins:**

| Category | Plugin | Key output |
|---|---|---|
| Structural | `rmsd` | Per-frame RMSD vs. reference |
| Structural | `rmsf` | Per-residue RMSF |
| Structural | `radius_of_gyration` | Per-frame Rg |
| Structural | `secondary_structure` | DSSP time evolution |
| Interaction | `hbond` | Donor-acceptor pairs, occupancy |
| Interaction | `salt_bridges` | Persistent salt bridge pairs |
| Interaction | `contacts` | Residue-residue contact map |
| Thermodynamic | `mmpbsa` | Per-residue energy decomposition |
| Thermodynamic | `sasa` | Per-frame SASA for user-specified residue(s) |
| Dynamic | `pca` | PC1/PC2 projections, explained variance |
| Dynamic | `dccm` | Dynamic cross-correlation matrix |

All plugins share a single MDAnalysis `Universe` constructed once from `prod.xtc` + `prod.tpr`.

**Outputs:**
- `output/stage4/<plugin_name>.csv` — raw numeric data
- `output/stage4/<plugin_name>.png` — matplotlib figure
- `output/stage4/report.html` — Jinja2-rendered summary with all figures and statistics

---

## 6. Master Config Schema

```yaml
input:
  source: pdb                          # pdb | uniprot
  path: protein.pdb                    # path if source=pdb
  uniprot_id: P12345                   # used if source=uniprot
  ph: 7.4                              # protonation pH

modification:
  # prediction_mode is NOT used during `phosp run`.
  # Run `phosp predict-sites config.yaml` separately to generate site suggestions.
  sites:
    - chain: A
      resid: 42
      resname: SER
      phospho_type: pSer               # pSer | pThr | pTyr

forcefield: charmm36m                  # charmm36m | amber_ff14sb
engine: gromacs
protocol: globular_protein             # named preset or path to custom YAML

simulation:
  production_time_ns: 100
  output_freq_ps: 10
  water_model: tip3p                   # tip3p | spce
  box_type: dodecahedron               # dodecahedron | cubic
  salt_concentration_mM: 150
  hpc:
    enabled: false
    scheduler: slurm                   # slurm | pbs
    ntasks: 8
    gpus: 1
    walltime: "24:00:00"
    partition: gpu

analysis:
  plugins: [rmsd, rmsf, hbond, contacts, pca]
  rmsd:
    selection: backbone
    reference: first_frame
  rmsf:
    selection: "name CA"
  mmpbsa:
    method: pbsa                       # pbsa | gbsa
    temperature: 300
  sasa:
    residues: [42, 87]                 # resids to track; empty = whole protein
```

---

## 7. CLI

```
phosp run config.yaml                          # full pipeline
phosp run config.yaml --start-from stage2      # resume from stage
phosp run config.yaml --stages 1,2             # run specific stages only
phosp predict-sites config.yaml                # run NetPhos, print suggested sites
phosp report output/                           # regenerate HTML report from existing results
phosp validate config.yaml                     # dry-run config validation only
```

---

## 8. Error Handling

- Each stage raises typed exceptions: `StageInputError`, `ModificationError`, `PreparationError`, `SimulationError`, `AnalysisError`
- Pipeline catches and logs with full context (stage, step, relevant file paths)
- GROMACS failures include the last 50 lines of the relevant `.log` file
- NetPhos unavailability degrades gracefully: prediction mode raises a clear error; config mode is unaffected

---

## 9. Testing Strategy

- **Unit tests:** Each modifier, force field class, and analysis plugin tested in isolation with fixture PDB files
- **Integration tests:** Full Stage 1 + Stage 2 run on a small test protein (e.g., Ubiquitin, 76 residues) with both CHARMM36m and AMBER ff14SB
- **Stage 3 smoke test:** Short 100 ps production run on CI (GPU not required — CPU fallback)
- **Plugin tests:** Each analysis plugin tested against a pre-computed trajectory fixture; output DataFrame schema validated
- **Config validation tests:** Invalid configs checked for correct error messages

---

## 10. Out of Scope (v1)

- Enhanced sampling methods (metadynamics, replica exchange)
- Membrane protein builder (CHARMM-GUI-style; the `membrane_protein` preset covers basic rectangular box setup only)
- Multi-protein batch orchestration (single protein per run in v1)
- Non-canonical phosphorylation (pHis, pAsp, pLys)
- GUI or web interface
