from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import MDAnalysis as mda
from matplotlib.figure import Figure
from phosp.plugins.analysis.base import AnalysisPlugin, PROTEIN_SELECTION


class RadiusOfGyrationPlugin(AnalysisPlugin):
    name = "radius_of_gyration"

    def run(self, universe: mda.Universe, config: dict) -> pd.DataFrame:
        protein = universe.select_atoms(PROTEIN_SELECTION)
        times, rg_values = [], []
        for ts in universe.trajectory:
            times.append(ts.time)
            rg_values.append(protein.radius_of_gyration())
        return pd.DataFrame({"time_ps": times, "rg_angstrom": rg_values})

    def plot(self, result: pd.DataFrame) -> Figure:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(result["time_ps"] / 1000, result["rg_angstrom"])
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Rg (Å)")
        ax.set_title("Radius of Gyration")
        fig.tight_layout()
        return fig
