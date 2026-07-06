from __future__ import annotations
from pathlib import Path
import typer

app = typer.Typer(
    help="Automated phosphorylation + MD simulation pipeline",
    no_args_is_help=True,
)

_STARTER_CONFIG = """\
# phosp configuration — run `phosp validate <this-file>` to check it.

input:
  source: pdb           # "pdb" or "uniprot"
  path: protein.pdb     # path to PDB file (required when source=pdb)
  # uniprot_id: P12345  # UniProt accession (required when source=uniprot)
  ph: 7.4               # pH for protonation state assignment

modification:
  sites:
    - chain: A          # PDB chain ID
      resid: 42         # residue number
      resname: SER      # must match mod_type: SER/THR/TYR for pSer/pThr/pTyr, LYS for acetylLys/methylLys1/2/3
      mod_type: pSer      # pSer, pThr, pTyr, acetylLys, methylLys1, methylLys2, or methylLys3

forcefield: charmm36m   # "charmm36m" or "amber_ff14sb"
protocol: globular_protein  # "globular_protein", "membrane_protein", or "phosphopeptide"

gromacs:
  binary: gmx           # gmx binary name or full path (e.g. "gmx_mpi", "/opt/gromacs/bin/gmx")
  # pdb2pqr: pdb2pqr   # pdb2pqr binary name or full path; default is "pdb2pqr"

simulation:
  production_time_ns: 100.0   # production run length in ns
  output_freq_ps: 10.0        # trajectory output frequency in ps
  water_model: tip3p          # "tip3p" or "spce"
  box_type: dodecahedron      # "dodecahedron" or "cubic"
  salt_concentration_mM: 150.0
  gpu_id: ~             # GPU index for mdrun (0, 1, …); ~ = GROMACS auto-select
  runner: local         # "local" | "slurm" | "pbs"

  hpc:                  # used when runner is slurm or pbs
    ntasks: 8           # OpenMP threads per rank (SLURM: cpus-per-task)
    gpus: 1
    walltime: "24:00:00"
    partition: gpu
    auto_submit: false
    # gromacs_module: gromacs/2026.0-cuda  # module to load; omit if GROMACS is already on PATH
    # extra_directives:                    # any additional scheduler options, e.g.:
    #   - "--account=myproject"
    #   - "--qos=high"
    #   - "--constraint=a100"
    #   - "--mem=128G"

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
    selection: name CA
  rmsf:
    selection: name CA
  sasa:
    residues: []

# Next: phosp predict-sites <this-file>
"""


def _version_callback(value: bool) -> None:
    if value:
        from importlib.metadata import version
        typer.echo(f"phosp {version('phosp')}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        None, "--version", "-V", callback=_version_callback, is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    pass


@app.command(help="Run the full pipeline (or resume from checkpoint).")
def run(
    config_path: Path = typer.Argument(..., help="Path to config YAML"),
    start_from: str = typer.Option(None, "--start-from", help="Resume from stage1|stage2|stage3|stage4"),
    stages: str = typer.Option(None, "--stages", help="Run only these stages, e.g. '1,2'"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Check config and tools without executing stages"),
    log_level: str = typer.Option("INFO", "--log-level", help="DEBUG|INFO|WARNING|ERROR"),
    log_file: Path = typer.Option(None, "--log-file", help="Write logs to this file"),
    reference: bool = typer.Option(False, "--reference", help="Run unmodified protein as reference (skips phosphorylation). Output goes to output_reference/."),
) -> None:
    from phosp.config import load_config
    from phosp.logging import configure_logging
    from phosp.pipeline import Pipeline
    from phosp.ui import PhospUI

    try:
        configure_logging(level=log_level, log_file=log_file)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    try:
        cfg = load_config(config_path)
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    output_root = config_path.parent / ("output_reference" if reference else "output")

    if dry_run:
        if cfg.input.source == "pdb" and cfg.input.path and not cfg.input.path.exists():
            typer.echo(f"Error: input PDB not found: {cfg.input.path}", err=True)
            raise typer.Exit(code=1)
        p = Pipeline(cfg, output_root=output_root, config_path=config_path, reference_mode=reference)
        p._preflight_checks()
        p.execute(dry_run=True, start_from=start_from, only_stages=stages)
        estimated_gb = cfg.simulation.production_time_ns * 1.0 + 0.5
        typer.echo(f"Estimated disk space needed: {estimated_gb:.1f} GB")
        typer.echo("Dry run complete — config and environment OK")
        return

    ui = PhospUI()
    Pipeline(cfg, output_root=output_root, ui=ui, config_path=config_path, reference_mode=reference).execute(
        start_from=start_from, only_stages=stages
    )


@app.command(help="Validate config syntax and check that required tools are available.")
def validate(
    config_path: Path = typer.Argument(..., help="Path to config YAML"),
) -> None:
    import shutil
    from phosp.config import load_config
    from phosp.logging import configure_logging
    from phosp.pipeline import Pipeline
    configure_logging()
    try:
        cfg = load_config(config_path)
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    errors: list[str] = []
    gmx = cfg.gromacs.binary
    if shutil.which(gmx) is None and not Path(gmx).is_file():
        errors.append(
            f"GROMACS binary '{gmx}' not found — "
            "install GROMACS or set gromacs.binary in the config"
        )
    pdb2pqr = cfg.gromacs.pdb2pqr
    if shutil.which(pdb2pqr) is None and not Path(pdb2pqr).is_file():
        errors.append(
            f"pdb2pqr binary '{pdb2pqr}' not found — "
            "run: pip install pdb2pqr  (or set gromacs.pdb2pqr in config)"
        )
    if cfg.input.source == "pdb" and cfg.input.path and not cfg.input.path.exists():
        errors.append(f"input PDB not found: {cfg.input.path}")

    if errors:
        for e in errors:
            typer.echo(f"  ✗ {e}", err=True)
        raise typer.Exit(code=1)

    # Run the FF-specific preflight (e.g. CHARMM36m directory check)
    try:
        p = Pipeline(cfg, output_root=config_path.parent / "output")
        p._check_forcefield()
    except Exception as exc:
        typer.echo(f"  ✗ {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo("  ✓ Config valid")
    typer.echo(f"  ✓ {gmx} found")
    typer.echo(f"  ✓ {pdb2pqr} found")
    typer.echo("  ✓ Force field ready")


@app.command(name="predict-sites", help="Predict phosphorylatable residues using NetPhos.")
def predict_sites(
    config_path: Path = typer.Argument(..., help="Path to config YAML"),
    threshold: float = typer.Option(0.5, "--threshold", help="Minimum score threshold (0–1)"),
) -> None:
    from phosp.config import load_config
    from phosp.logging import configure_logging
    from phosp.prediction.netphos import NetPhos
    configure_logging()
    try:
        cfg = load_config(config_path)
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    if cfg.input.path is None:
        typer.echo("Error: predict-sites requires input.source=pdb with a path set", err=True)
        raise typer.Exit(code=1)
    timeout_minutes = cfg.gromacs.timeout_minutes
    try:
        results = NetPhos().predict(
            cfg.input.path, threshold=threshold,
            timeout=timeout_minutes * 60 if timeout_minutes else None,
        )
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        typer.echo("NetPhos is not bundled with phosp. Download it from https://services.healthtech.dtu.dk/software.php and ensure 'netphos' is on your PATH.")
        raise typer.Exit(code=1)
    for r in results:
        typer.echo(f"  chain={r['chain']} resid={r['resid']} resname={r['resname']} "
                   f"type={r['mod_type']} score={r['score']:.3f}")
    typer.echo(f"\nAdd selected entries to modification.sites in {config_path}")


@app.command(help="Regenerate the HTML report from an existing stage4 output directory.")
def report(
    output_dir: Path = typer.Argument(..., help="Pipeline output directory (contains stage4/)"),
) -> None:
    from phosp.logging import configure_logging
    from phosp.stages.stage4_analyze import Stage4Analyze
    configure_logging()
    stage4_dir = output_dir / "stage4"
    if not stage4_dir.exists():
        typer.echo(f"Error: stage4 output not found at {stage4_dir}", err=True)
        typer.echo("Run 'phosp run <config>' to completion first, then re-run this command.")
        raise typer.Exit(code=1)
    Stage4Analyze.regenerate_report(stage4_dir)
    typer.echo(f"Report written to {stage4_dir / 'report.html'}")


@app.command(help="Write a starter config file to get started quickly.")
def init(
    path: Path = typer.Argument(Path("phosp_config.yaml"), help="Output path for the config file"),
) -> None:
    from phosp.logging import configure_logging
    configure_logging()
    if path.exists():
        typer.echo(f"Error: {path} already exists. Use a different path or delete it first.", err=True)
        raise typer.Exit(code=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_STARTER_CONFIG)
    typer.echo(f"Config written to {path}")
    typer.echo(f"Next steps:")
    typer.echo(f"  1. Edit {path} (set input.path, modification.sites, production_time_ns)")
    typer.echo(f"  2. phosp validate {path}")
    typer.echo(f"  3. phosp run {path}")


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 1:
        return "< 1 s"
    if seconds < 60:
        return f"{int(seconds)} s"
    mins, secs = divmod(int(seconds), 60)
    if mins < 60:
        return f"{mins} m {secs} s"
    hours, mins = divmod(mins, 60)
    return f"{hours} h {mins} m"


@app.command(help="Show completed stages and checkpoint state for a pipeline run.")
def status(
    output_dir: Path = typer.Argument(..., help="Pipeline output directory (contains checkpoint.json)"),
) -> None:
    import json
    from rich.console import Console
    from rich.table import Table
    from phosp.logging import configure_logging
    configure_logging()

    checkpoint_path = output_dir / "checkpoint.json"
    if not checkpoint_path.exists():
        typer.echo(f"Error: no checkpoint found at {checkpoint_path}", err=True)
        typer.echo(f"Tip: use <config-dir>/output/ for normal runs, <config-dir>/output_reference/ for --reference runs.")
        raise typer.Exit(code=1)

    data = json.loads(checkpoint_path.read_text())
    completed = set(data.get("completed_stages", []))

    _labels = {
        "stage1": "Stage 1 — Chemical Modification",
        "stage2": "Stage 2 — MD Preparation",
        "stage3": "Stage 3 — MD Simulation",
        "stage4": "Stage 4 — Analysis",
    }

    console = Console()
    table = Table(title="phosp Pipeline Status")
    table.add_column("Stage", style="cyan", no_wrap=True)
    table.add_column("Status")
    table.add_column("Completed At")
    table.add_column("Key Artifacts")
    table.add_column("Duration")

    for s in ["stage1", "stage2", "stage3", "stage4"]:
        if s in completed:
            status_str = "[green]✓ complete[/]"
            completed_at = data.get(f"{s}_completed_at", "")
            artifacts = data.get("artifacts", {}).get(s, {})
            artifact_str = ", ".join(Path(v).name for v in list(artifacts.values())[:3])
            duration_str = _fmt_duration(data.get(f"{s}_duration_seconds"))
        else:
            status_str = "[dim]pending[/]"
            completed_at = ""
            artifact_str = ""
            duration_str = "—"
        table.add_row(_labels[s], status_str, completed_at, artifact_str, duration_str)

    console.print(table)

    if not {"stage1", "stage2", "stage3", "stage4"}.issubset(completed):
        raise typer.Exit(code=1)


@app.command(help="Remove pipeline output directory and checkpoint (with confirmation).")
def clean(
    output_dir: Path = typer.Argument(..., help="Pipeline output directory to remove (e.g. output/)"),
) -> None:
    import shutil
    from phosp.logging import configure_logging
    configure_logging()

    if not output_dir.exists():
        typer.echo(f"Error: output directory not found: {output_dir}", err=True)
        raise typer.Exit(code=1)
    if not output_dir.is_dir():
        typer.echo(f"Error: not a directory: {output_dir}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Will remove: {output_dir.resolve()}")
    typer.confirm("Proceed? This cannot be undone.", abort=True)

    shutil.rmtree(output_dir)
    typer.echo(f"Removed {output_dir}")
