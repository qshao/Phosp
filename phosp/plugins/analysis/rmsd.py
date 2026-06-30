from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import rms
from matplotlib.figure import Figure
from phosp.plugins.analysis.base import AnalysisPlugin


class RMSDPlugin(AnalysisPlugin):
    name = "rmsd"

    def run(self, universe: mda.Universe, config: dict) -> pd.DataFrame:
        selection = config.get("selection", "backbone")
        universe.trajectory[0]  # reset to frame 0 so copy() captures it as reference
        reference = universe.copy()
        R = rms.RMSD(universe, reference, select=selection)
        R.run()
        return pd.DataFrame({
            "frame": R.results.rmsd[:, 0].astype(int),
            "time_ps": R.results.rmsd[:, 1],
            "rmsd_angstrom": R.results.rmsd[:, 2],
        })

    def plot(self, result: pd.DataFrame) -> Figure:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(result["time_ps"] / 1000, result["rmsd_angstrom"])
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("RMSD (Å)")
        ax.set_title("Backbone RMSD")
        fig.tight_layout()
        return fig
