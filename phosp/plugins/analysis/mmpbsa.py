from __future__ import annotations
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import MDAnalysis as mda
from matplotlib.figure import Figure
from phosp.plugins.analysis.base import AnalysisPlugin


class MMPBSAPlugin(AnalysisPlugin):
    name = "mmpbsa"

    def run(self, universe: mda.Universe, config: dict) -> pd.DataFrame:
        exe = shutil.which("gmx_MMPBSA")
        if not exe:
            raise RuntimeError(
                "gmx_MMPBSA not found in PATH. "
                "Install it with: pip install gmx_MMPBSA"
            )
        method = config.get("method", "pbsa")
        temperature = config.get("temperature", 300)

        traj = getattr(universe.trajectory, "filename", None)
        top = getattr(universe, "filename", None)
        if not traj or not top:
            raise RuntimeError("Cannot determine trajectory/topology paths from Universe.")

        # Write scratch/output files into this analysis stage's own directory
        # (passed by stage4_analyze.py), not Path(traj).parent — that's a
        # previous, already-finalized stage's output directory (stage3's
        # production/), and gmx_MMPBSA's numerous intermediate files would
        # otherwise pollute it on every stage4 run.
        work_dir = Path(config.get("_work_dir") or Path(traj).parent)
        work_dir.mkdir(parents=True, exist_ok=True)
        input_file = work_dir / "mmpbsa.in"
        input_file.write_text(
            f"&general\n"
            f"  startframe=1, endframe=99999, interval=1,\n"
            f"  temperature={temperature},\n"
            f"/\n"
            f"&{method}\n/\n"
            f"&decomp\n  idecomp=2,\n/\n"
        )
        cmd = [
            exe,
            "-O",
            "-i", str(input_file),
            "-cs", str(top),
            "-ct", str(traj),
            "-cp", str(Path(traj).parent.parent.parent / "stage2" / "topol.top"),
            "-o", str(work_dir / "mmpbsa_results.dat"),
            "-eo", str(work_dir / "mmpbsa_energies.csv"),
        ]
        timeout_minutes = config.get("_timeout_minutes")
        timeout = timeout_minutes * 60 if timeout_minutes else None
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"gmx_MMPBSA timed out after {timeout_minutes} minutes")
        if result.returncode != 0:
            raise RuntimeError(f"gmx_MMPBSA failed:\n{result.stderr[-2000:]}")

        csv_path = work_dir / "mmpbsa_energies.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return pd.DataFrame(columns=["Residue", "Total"])

    def plot(self, result: pd.DataFrame) -> Figure:
        fig, ax = plt.subplots(figsize=(10, 4))
        if "Residue" in result.columns and "Total" in result.columns:
            ax.bar(result["Residue"].astype(str), result["Total"])
            ax.set_xlabel("Residue")
            ax.set_ylabel("ΔG (kcal/mol)")
            ax.set_title("MM-PBSA Per-residue Energy Decomposition")
            plt.xticks(rotation=90, fontsize=7)
        else:
            ax.text(0.5, 0.5, "MM-PBSA data unavailable", ha="center", va="center")
        fig.tight_layout()
        return fig
