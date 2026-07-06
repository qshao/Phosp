from __future__ import annotations
import importlib
import logging
import pkgutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda

import phosp.plugins.analysis as _analysis_pkg
from phosp.exceptions import AnalysisError, StageInputError
from phosp.plugins.analysis.base import AnalysisPlugin
from phosp.stages.base import Stage, StageResult

logger = logging.getLogger(__name__)


def _discover_plugins() -> dict[str, type[AnalysisPlugin]]:
    registry: dict[str, type[AnalysisPlugin]] = {}
    pkg_path = Path(_analysis_pkg.__file__).parent
    for _, mod_name, _ in pkgutil.iter_modules([str(pkg_path)]):
        if mod_name == "base":
            continue
        importlib.import_module(f"phosp.plugins.analysis.{mod_name}")
    for cls in AnalysisPlugin.__subclasses__():
        registry[cls.name] = cls
    return registry


class Stage4Analyze(Stage):
    def validate_inputs(self) -> None:
        prod_dir = self.output_root.parent / "stage3" / "production"
        pending_sentinel = self.output_root.parent / "stage3" / "pending_job.json"
        for f in ["production.xtc", "production.gro"]:
            if not (prod_dir / f).exists():
                if pending_sentinel.exists():
                    import json as _json
                    info = _json.loads(pending_sentinel.read_text())
                    sched = info.get("scheduler", "hpc").upper()
                    job_id = info.get("job_id")
                    check_cmd = "squeue" if sched == "SLURM" else "qstat"
                    id_part = f" (job {job_id})" if job_id else ""
                    raise StageInputError(
                        f"{sched} job{id_part} is still running — {f} not yet available.\n"
                        f"  Check status:  {check_cmd}\n"
                        "  Re-run 'phosp run <config>' after the job completes."
                    )
                raise StageInputError(
                    f"{f} not found in {prod_dir}. Run stage3 first."
                )

    def run(self) -> StageResult:
        out = self.output_root
        out.mkdir(parents=True, exist_ok=True)
        cfg = self.config
        prod_dir = out.parent / "stage3" / "production"

        registry = _discover_plugins()
        requested = cfg.analysis.plugins
        unknown = [p for p in requested if p not in registry]
        if unknown:
            raise AnalysisError(
                f"Unknown analysis plugins: {unknown}. "
                f"Valid plugins: {sorted(registry)}"
            )

        # Use GRO as topology — avoids TPR version incompatibilities with MDAnalysis
        universe = mda.Universe(
            str(prod_dir / "production.gro"),
            str(prod_dir / "production.xtc"),
        )

        artifacts: dict[str, Path] = {}
        failures: list[tuple[str, str]] = []

        for plugin_name in requested:
            plugin_config = getattr(cfg.analysis, plugin_name, {})
            if not isinstance(plugin_config, dict):
                plugin_config = plugin_config.model_dump() if hasattr(plugin_config, "model_dump") else {}
            # Lets subprocess-based plugins (e.g. mmpbsa) bound external tool
            # runtime and write scratch/output files into this stage's own
            # output directory instead of a previous (finalized) stage's.
            plugin_config = {
                **plugin_config,
                "_timeout_minutes": cfg.gromacs.timeout_minutes,
                "_work_dir": out,
            }

            if self.ui:
                self.ui.plugin_start(plugin_name)

            try:
                plugin = registry[plugin_name]()
                result_df = plugin.run(universe, plugin_config)
                csv_path = out / f"{plugin_name}.csv"
                result_df.to_csv(csv_path, index=False)

                fig = plugin.plot(result_df)
                png_path = out / f"{plugin_name}.png"
                fig.savefig(png_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

                artifacts[plugin_name] = csv_path
                logger.info("Plugin '%s' complete", plugin_name)
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                logger.warning("Plugin '%s' failed: %s", plugin_name, error_msg)
                failures.append((plugin_name, error_msg))

        final_out = out.parent / out.name.lstrip('.').removesuffix('_tmp') if out.name.startswith('.') else out
        if failures and not artifacts:
            _render_report(out, report_dir=final_out, failed_plugins=failures)
            raise AnalysisError(
                "All analysis plugins failed:\n"
                + "\n".join(f"  {n}: {e}" for n, e in failures)
            )

        _render_report(out, report_dir=final_out, failed_plugins=failures)
        return StageResult(stage="stage4", output_dir=out, artifacts=artifacts)

    @staticmethod
    def regenerate_report(output_dir: Path) -> None:
        _render_report(output_dir, failed_plugins=[])


def _render_report(
    output_dir: Path,
    report_dir: Path | None = None,
    failed_plugins: list[tuple[str, str]] | None = None,
) -> Path:
    from jinja2 import Environment, FileSystemLoader
    import base64
    templates_dir = Path(__file__).parent.parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=True,
    )
    template = env.get_template("report.html.j2")

    png_files = sorted(output_dir.glob("*.png"))
    figures = []
    for png in png_files:
        b64 = base64.b64encode(png.read_bytes()).decode()
        figures.append({"name": png.stem, "data": b64})

    html = template.render(
        figures=figures,
        output_dir=str(report_dir or output_dir),
        failed_plugins=failed_plugins or [],
    )
    report_path = output_dir / "report.html"
    report_path.write_text(html)
    return report_path
