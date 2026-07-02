# Installation guide

This guide covers every dependency in detail. If you just want the quick version, follow the [README](../README.md#installation).

---

## Table of contents

1. [System requirements](#system-requirements)
2. [Python virtual environment](#python-virtual-environment)
   - [conda (recommended)](#conda-recommended)
   - [venv](#venv)
3. [GROMACS](#gromacs)
   - [Via conda (easiest)](#via-conda-easiest)
   - [Via package manager](#via-package-manager)
   - [Compiled from source](#compiled-from-source)
4. [pdb2pqr](#pdb2pqr)
5. [CHARMM36m force field](#charmm36m-force-field)
6. [phosp itself](#phosp-itself)
7. [Verifying everything works](#verifying-everything-works)
8. [Platform notes](#platform-notes)
9. [HPC environments](#hpc-environments)

---

## System requirements

| Item | Requirement |
|---|---|
| Operating system | Linux (any), macOS 12+, Windows 10+ (WSL2 recommended) |
| Python | 3.10, 3.11, or 3.12 |
| RAM | ≥ 8 GB (16 GB+ for large proteins) |
| Disk | ≥ 20 GB free (100 ns trajectory ≈ 10–30 GB depending on protein size) |
| CPU | Any x86-64 or ARM64 |
| GPU | Optional — GROMACS uses CUDA or OpenCL automatically when available |

---

## Python virtual environment

Always work inside a virtual environment. This isolates phosp's dependencies from your system Python and from other projects.

### conda (recommended)

conda can install GROMACS alongside Python in the same environment, which is the most friction-free setup.

```bash
# Install Miniconda if you don't have it
# https://docs.conda.io/en/latest/miniconda.html
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create environment with Python 3.12
conda create -n phosp python=3.12 -y
conda activate phosp

# Install GROMACS from conda-forge (see GROMACS section)
conda install -c conda-forge gromacs -y
```

To activate this environment in future sessions:

```bash
conda activate phosp
```

To deactivate:

```bash
conda deactivate
```

### venv

Use venv when conda is not available or when you have GROMACS installed separately via your package manager or from source.

```bash
# Create the environment
python3.12 -m venv ~/envs/phosp

# Activate — Linux / macOS
source ~/envs/phosp/bin/activate

# Activate — Windows (Command Prompt)
%USERPROFILE%\envs\phosp\Scripts\activate.bat

# Activate — Windows (PowerShell)
~\envs\phosp\Scripts\Activate.ps1
```

Your prompt should now show `(phosp)`. To deactivate later:

```bash
deactivate
```

---

## GROMACS

### Via conda (easiest)

If you created a conda environment as above, install GROMACS from conda-forge.

```bash
conda install -c conda-forge gromacs -y
```

Verify:

```bash
gmx --version
```

**GPU support caveat:** `gmx --version` prints a "GPU support" line — check it before assuming your run will use the GPU. conda-forge's GROMACS ships with different GPU backends depending on platform, and the backend has to match your hardware:

- On **linux-64**, conda-forge provides a CUDA-enabled build for NVIDIA GPUs. This is the normal case for datacenter GPUs like **A100, H100, and H200** — they're always attached to x86-64 (linux-64) hosts, so `conda install -c conda-forge gromacs` on such a node picks up the CUDA build automatically (conda-forge detects the local driver via the `__cuda` virtual package). No special flags needed. After installing, confirm with `gmx --version` — look for `GPU support: CUDA` and a build targeting your GPU's compute capability (A100 = Ampere/sm_80, H100/H200 = Hopper/sm_90; recent GROMACS builds bundle both).
- On **linux-aarch64** (ARM64, e.g. Jetson, Grace-Blackwell-class systems), conda-forge currently ships an **OpenCL** build only — there is no CUDA variant. OpenCL does not support newer NVIDIA GPU generations (Volta and later); `gmx mdrun` will silently fall back to CPU-only and run 10-50x slower with no error, only a line in the log like `status: incompatible (please use CUDA build for NVIDIA Volta GPUs or newer)`. Confirm actual GPU use by checking for a `Performance: X ns/day` line with realistic throughput in the `mdrun` log, not just that the run started.
- Getting real GPU acceleration in that situation means compiling GROMACS from source with `-DGMX_GPU=CUDA` (see below) — it needs the CUDA toolkit installed and takes 10–30 minutes.

**Getting phosp to actually use the GPU:** this depends on `simulation.runner`:

- Running locally (`runner: local`): set `simulation.gpu_id` (e.g. `0`) explicitly. phosp only adds the `-nb gpu -pme gpu -bonded gpu -update gpu` offload flags to `mdrun` when `gpu_id` is set, so a run with `gpu_id: ~` still works but under-uses a fast GPU (only nonbonded work is offloaded; PME, bonded terms, and the integrator/constraint update stay on CPU).
- Running on a cluster (`runner: slurm` or `runner: pbs`): leave `gpu_id: ~` — you can't know which GPU/node SLURM or PBS will assign until the job starts — and set `hpc.gpus: 1` instead (the default). The offload flags are added automatically whenever a GPU is requested (`hpc.gpus > 0`), so whatever device the scheduler hands you gets fully used, no device index required.

`-update gpu` is skipped for the minimization phase specifically (it requires a dynamical integrator; minimization uses `steep`, which GROMACS rejects with `-update gpu`) — `-nb`/`-pme`/`-bonded gpu` still apply there, and NVT/NPT/production get the full flag set.

See the "Using a datacenter GPU" note in the [README](../README.md#simulation-block) for full detail, and the [HPC environments](#hpc-environments) section below for setting up phosp on a cluster where GROMACS/CUDA come from environment modules or a fixed install path.

### Via package manager

**Ubuntu / Debian:**

```bash
sudo apt-get update
sudo apt-get install gromacs
```

This installs an older GROMACS (often 2022 or earlier). It works with phosp but a newer version is preferred.

**Homebrew (macOS):**

```bash
brew install gromacs
```

### Compiled from source

Compiling from source gives the newest version and lets you enable GPU acceleration. This takes 10–30 minutes.

```bash
# Prerequisites
sudo apt-get install cmake build-essential libfftw3-dev    # Ubuntu/Debian

# Download GROMACS 2026.0 (or the latest release)
wget https://ftp.gromacs.org/gromacs/gromacs-2026.0.tar.gz
tar -xzf gromacs-2026.0.tar.gz
cd gromacs-2026.0

mkdir build && cd build
cmake .. \
  -DGMX_BUILD_OWN_FFTW=ON \
  -DREGRESSIONTEST_DOWNLOAD=ON \
  -DCMAKE_INSTALL_PREFIX=/usr/local/gromacs
make -j$(nproc)
sudo make install

# Add to PATH
echo 'source /usr/local/gromacs/bin/GMXRC' >> ~/.bashrc
source ~/.bashrc
```

**With CUDA GPU support:**

```bash
cmake .. \
  -DGMX_BUILD_OWN_FFTW=ON \
  -DGMX_GPU=CUDA \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
  -DCMAKE_INSTALL_PREFIX=/usr/local/gromacs
```

**With OpenCL GPU support (AMD / Intel):**

```bash
cmake .. \
  -DGMX_BUILD_OWN_FFTW=ON \
  -DGMX_GPU=OpenCL \
  -DCMAKE_INSTALL_PREFIX=/usr/local/gromacs
```

If you compile with a non-default binary name (e.g. `gmx_mpi`), set it in phosp's config:

```yaml
gromacs:
  binary: gmx_mpi
```

---

## pdb2pqr

pdb2pqr handles protonation state assignment at a chosen pH using PROPKA.

```bash
pip install pdb2pqr
```

Verify:

```bash
pdb2pqr --version
# should print 3.x.x
```

If the command is not found after installation, your virtual environment's `bin/` directory may not be on `PATH`. With a venv that is active, this should not happen. With conda:

```bash
conda run -n phosp pip install pdb2pqr
```

---

## CHARMM36m force field

GROMACS does not ship the CHARMM36m force field. You must download and install it manually. This is a one-time setup per GROMACS installation.

### Step 1 — Download from MacKerell lab

```bash
curl -L "https://mackerell.umaryland.edu/download.php?filename=CHARMM_ff_params_files/charmm36-jul2022.ff.tgz" \
     -o charmm36-jul2022.ff.tgz
```

> The publicly available file is `charmm36-jul2022.ff.tgz` (without the `m` suffix). The `m` variant is not separately distributed; the standard CHARMM36-jul2022 parameters are equivalent for protein simulations.

### Step 2 — Find the GROMACS topology directory

```bash
GMXTOP=$(gmx --version 2>&1 | awk '/Data prefix/{print $3}')/share/gromacs/top
echo "GROMACS topology directory: $GMXTOP"
```

This works for any GROMACS installation (package manager, conda, or compiled). Double-check that the directory exists:

```bash
ls "$GMXTOP" | grep -E "charmm|amber|tip"
```

### Step 3 — Extract the force field

```bash
tar -xzf charmm36-jul2022.ff.tgz -C "$GMXTOP"
ls "$GMXTOP/charmm36-jul2022.ff/"    # should list many .itp files
```

### Step 4 — Create the symlink phosp expects

phosp looks for `charmm36m-jul2022.ff`. Create a symlink pointing at the extracted directory:

```bash
ln -s "$GMXTOP/charmm36-jul2022.ff" "$GMXTOP/charmm36m-jul2022.ff"
```

### Step 5 — Register phospho-residue types

pdb2gmx refuses to process residues it doesn't recognise as `Protein`. Tell GROMACS that SEP, TPO, and PTR are protein residues:

```bash
printf "SEP\tProtein\nTPO\tProtein\nPTR\tProtein\n" >> "$GMXTOP/residuetypes.dat"
```

Verify:

```bash
grep -E "SEP|TPO|PTR" "$GMXTOP/residuetypes.dat"
# should print three lines
```

### Step 6 — Fix a naming collision in the ether terminal database

A residue name `MET1` in `ethers.n.tdb` collides with a protein methionine terminal definition. Rename it:

```bash
sed -i 's/^\[ MET1 \]/[ EMETH1 ]/' "$GMXTOP/charmm36-jul2022.ff/ethers.n.tdb"
```

Verify the fix applied:

```bash
grep "EMETH1" "$GMXTOP/charmm36-jul2022.ff/ethers.n.tdb"
```

---

## phosp itself

```bash
git clone https://github.com/qshao/Phosp.git
cd Phosp
pip install -e .
```

The `-e` flag installs in editable mode, meaning changes to the source are reflected immediately without reinstalling.

For development work (adds `pytest`, `ruff`):

```bash
pip install -e ".[dev]"
```

---

## Verifying everything works

Run the bundled validation command against the ubiquitin example:

```bash
phosp validate examples/ubiquitin_pThr/run.yaml
```

Expected output:

```
  ✓ Config valid
  ✓ gmx found
  ✓ pdb2pqr found
  ✓ Force field ready
```

Then run the test suite to confirm the Python layer is intact:

```bash
pytest
# 167 passed
```

---

## Platform notes

### macOS

GROMACS via Homebrew works but may lag a release or two behind. GPU support requires Metal compute shaders (GROMACS 2023+). For Apple Silicon (M1/M2/M3):

```bash
conda install -c conda-forge gromacs -y   # ARM-native build via conda-forge
```

### Windows

Native Windows is not recommended. Use WSL2:

1. Install WSL2: `wsl --install -d Ubuntu-22.04` (in an elevated PowerShell)
2. Open the Ubuntu terminal
3. Follow the Linux installation steps above

Alternatively, run phosp inside a Docker container (see below).

### Docker

```bash
docker pull continuumio/miniconda3
docker run -it --rm -v $(pwd):/workspace continuumio/miniconda3 bash

# Inside the container:
conda create -n phosp python=3.12 -y && conda activate phosp
conda install -c conda-forge gromacs -y
pip install pdb2pqr
# Follow CHARMM36m setup...
# Clone and pip install phosp...
```

---

## HPC environments

On most HPC clusters, GROMACS is provided as a module and is not available by default. phosp stages 1, 2, and 4 do not need GROMACS at runtime — only stage 3 does. This means you can run stages 1–2 and 4 on a login node.

### Typical workflow on an HPC cluster

**On the login node** — set up the environment once:

```bash
module load python/3.12          # or use your cluster's Python module
python -m venv ~/envs/phosp
source ~/envs/phosp/bin/activate
pip install -e /path/to/Phosp

# Install CHARMM36m pointing at the cluster's GROMACS topology directory
module load gromacs/2026
GMXTOP=$(gmx --version 2>&1 | awk '/Data prefix/{print $3}')/share/gromacs/top
# ... follow steps 1–6 above
```

**In your config**, use the MPI binary and set `runner: slurm`:

```yaml
gromacs:
  binary: gmx_mpi        # check what your cluster provides

simulation:
  runner: slurm
  hpc:
    ntasks: 32
    gpus: 1
    walltime: "48:00:00"
    partition: gpu
    auto_submit: false
```

**Run stages 1–3 to generate the job script:**

```bash
phosp run my_run/config.yaml --stages 1,2,3
sbatch my_run/output/stage3/run_slurm.sh
```

**After the job completes, run stage 4:**

```bash
phosp run my_run/config.yaml --stages 4
```

The GROMACS module does not need to be loaded for stage 4 — MDAnalysis reads the trajectory directly.

### GPU nodes: CUDA via module, GROMACS at a fixed path

A common cluster setup: CUDA is provided as an environment module, but GROMACS itself was compiled to a fixed install path rather than exposed as its own module — and you won't know which node or GPU SLURM/PBS assigns until the job actually starts. None of that requires special handling; it's already the normal case for this pipeline:

**Python virtual environment** — same as any HPC setup, on the login node:

```bash
module load python/3.12
python -m venv ~/envs/phosp
source ~/envs/phosp/bin/activate
pip install -e /path/to/Phosp
```

You do not need conda or a GROMACS package inside this venv — GROMACS at runtime comes from whatever `gromacs.binary` and `hpc.gromacs_module` you configure below, resolved on the compute node when the job runs, not on the login node.

**Config** — point `gromacs.binary` at the fixed install path, and use `hpc.gromacs_module` to load only the CUDA module (space-separate multiple module names if you need more than one):

```yaml
gromacs:
  binary: /shared/apps/gromacs-2026-cuda/bin/gmx_mpi

simulation:
  gpu_id: ~                 # leave unset — the assigned GPU's index isn't known in advance
  runner: slurm
  hpc:
    ntasks: 32
    gpus: 1                 # requesting >0 GPUs is what triggers full GPU offload flags
    walltime: "48:00:00"
    partition: gpu
    auto_submit: false
    gromacs_module: "cuda/12.4"   # only CUDA needs to come from the module system here
```

Because `gromacs.binary` is an absolute path, the generated job script invokes it directly — it doesn't need to be on `$PATH` or exposed as its own module. `hpc.gpus: 1` is what causes phosp to add the full offload flags (`-nb gpu -pme gpu -bonded gpu -update gpu`) to `mdrun`; this doesn't depend on knowing which physical GPU or node the scheduler assigns, since SLURM/PBS scope GPU visibility to the job regardless. Run the same three-step workflow above (stages 1-2 on the login node, stage 3 to submit, stage 4 after completion) — see the [HPC usage](../README.md#hpc-usage-slurm--pbs) section of the README for the full walkthrough.

### Identifying the correct binary name

```bash
module load gromacs/2026
which gmx       # if this prints a path, use "gmx"
which gmx_mpi   # MPI-enabled build; use "gmx_mpi" for parallel runs
```

Set whichever name is available in `gromacs.binary` in your config.
