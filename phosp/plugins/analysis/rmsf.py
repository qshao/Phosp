from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import rms
from matplotlib.figure import Figure
from phosp.plugins.analysis.base import AnalysisPlugin


class RMSFPlugin(AnalysisPlugin):
    name = "rmsf"

    def run(self, universe: mda.Universe, config: dict) -> pd.DataFrame:
        selection = config.get("selection", "name CA")
        atoms = universe.select_atoms(selection)
        R = rms.RMSF(atoms).run()
        return pd.DataFrame({
            "resid": atoms.resids,
            "resname": atoms.resnames,
            "rmsf_angstrom": R.results.rmsf,
        })

    def plot(self, result: pd.DataFrame) -> Figure:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(result["resid"], result["rmsf_angstrom"])
        ax.set_xlabel("Residue ID")
        ax.set_ylabel("RMSF (Å)")
        ax.set_title("Per-residue RMSF (Cα)")
        fig.tight_layout()
        return fig
