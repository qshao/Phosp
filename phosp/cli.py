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
      resname: SER      # SER, THR, or TYR
      phospho_type: pSer  # pSer, pThr, or pTyr (must match resname)

forcefield: charmm36m   # "charmm36m" or "amber_ff14sb"
protocol: globular_protein  # "globular_protein", "membrane_protein", or "phosphopeptide"

simulation:
  production_time_ns: 100.0   # production run length in ns
  output_freq_ps: 10.0        # trajectory output frequency in ps
  water_model: tip3p          # "tip3p" or "spce"
  box_type: dodecahedron      # "dodecahedron" or "cubic"
  salt_concentration_mM: 150.0

  hpc:
    enabled: false        # true to generate HPC job scripts
    scheduler: slurm      # "slurm" or "pbs"
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
    cfg = load_config(config_path)

    if dry_run:
        if cfg.input.source == "pdb" and cfg.input.path and not cfg.input.path.exists():
            typer.echo(f"Error: input PDB not found: {cfg.input.path}", err=True)
            raise typer.Exit(code=1)
        output_root = config_path.parent / "output"
        p = Pipeline(cfg, output_root=output_root)
        p._preflight_checks()
        p.execute(dry_run=True, start_from=start_from, only_stages=stages)
        estimated_gb = cfg.simulation.production_time_ns * 1.0 + 0.5
        typer.echo(f"Estimated disk space needed: {estimated_gb:.1f} GB")
        typer.echo("Dry run complete — config and environment OK")
        return

    ui = PhospUI()
    Pipeline(cfg, output_root=config_path.parent / "output", ui=ui).execute(
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
    cfg = load_config(config_path)

    errors: list[str] = []
    if shutil.which("gmx") is None:
        errors.append("gmx not found in PATH — install GROMACS")
    if shutil.which("pdb2pqr") is None:
        errors.append("pdb2pqr not found in PATH — run: pip install pdb2pqr")
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
    typer.echo("  ✓ gmx found")
    typer.echo("  ✓ pdb2pqr found")
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
    cfg = load_config(config_path)
    if cfg.input.path is None:
        typer.echo("Error: predict-sites requires input.source=pdb with a path set", err=True)
        raise typer.Exit(code=1)
    results = NetPhos().predict(cfg.input.path, threshold=threshold)
    for r in results:
        typer.echo(f"  chain={r['chain']} resid={r['resid']} resname={r['resname']} "
                   f"type={r['phospho_type']} score={r['score']:.3f}")
    typer.echo(f"\nAdd selected entries to modification.sites in {config_path}")


@app.command(help="Regenerate the HTML report from an existing stage4 output directory.")
def report(
    output_dir: Path = typer.Argument(..., help="Pipeline output directory (contains stage4/)"),
) -> None:
    from phosp.logging import configure_logging
    from phosp.stages.stage4_analyze import Stage4Analyze
    configure_logging()
    Stage4Analyze.regenerate_report(output_dir)
    typer.echo(f"Report written to {output_dir}/stage4/report.html")


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
        typer.echo(f"Tip: the output directory is usually <config-dir>/output/")
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

    for s in ["stage1", "stage2", "stage3", "stage4"]:
        if s in completed:
            status_str = "[green]✓ complete[/]"
            completed_at = data.get(f"{s}_completed_at", "")
            artifacts = data.get("artifacts", {}).get(s, {})
            artifact_str = ", ".join(Path(v).name for v in list(artifacts.values())[:3])
        else:
            status_str = "[dim]pending[/]"
            completed_at = ""
            artifact_str = ""
        table.add_row(_labels[s], status_str, completed_at, artifact_str)

    console.print(table)

    if not {"stage1", "stage2", "stage3", "stage4"}.issubset(completed):
        raise typer.Exit(code=1)
