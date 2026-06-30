from __future__ import annotations
from pathlib import Path
import typer

app = typer.Typer(help="Automated phosphorylation + MD simulation pipeline")


@app.command()
def run(
    config_path: Path = typer.Argument(..., help="Path to config YAML"),
    start_from: str = typer.Option(None, "--start-from", help="stage1|stage2|stage3|stage4"),
    stages: str = typer.Option(None, "--stages", help="e.g. '1,2'"),
) -> None:
    from phosp.config import load_config
    from phosp.pipeline import Pipeline
    cfg = load_config(config_path)
    Pipeline(cfg, output_root=config_path.parent / "output").execute(
        start_from=start_from, only_stages=stages
    )


@app.command()
def validate(config_path: Path = typer.Argument(...)) -> None:
    from phosp.config import load_config
    load_config(config_path)
    typer.echo("Config valid.")


@app.command(name="predict-sites")
def predict_sites(
    config_path: Path = typer.Argument(...),
    threshold: float = typer.Option(0.5, "--threshold"),
) -> None:
    from phosp.config import load_config
    from phosp.prediction.netphos import NetPhos
    cfg = load_config(config_path)
    pdb_path = cfg.input.path
    results = NetPhos().predict(pdb_path, threshold=threshold)
    for r in results:
        typer.echo(f"  chain={r['chain']} resid={r['resid']} resname={r['resname']} "
                   f"type={r['phospho_type']} score={r['score']:.3f}")
    typer.echo(f"\nAdd selected entries to modification.sites in {config_path}")


@app.command()
def report(output_dir: Path = typer.Argument(...)) -> None:
    from phosp.stages.stage4_analyze import Stage4Analyze
    Stage4Analyze.regenerate_report(output_dir)
    typer.echo(f"Report written to {output_dir}/report.html")
