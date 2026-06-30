from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
from matplotlib.figure import Figure
from phosp.plugins.analysis.base import AnalysisPlugin


class ContactsPlugin(AnalysisPlugin):
    name = "contacts"

    def run(self, universe: mda.Universe, config: dict) -> pd.DataFrame:
        cutoff = config.get("cutoff_angstrom", 8.0)
        selection = config.get("selection", "name CA")
        atoms = universe.select_atoms(selection)
        rows = []
        for ts in universe.trajectory:
            dist = distance_array(atoms.positions, atoms.positions)
            n_contacts = int(np.sum(dist < cutoff) - len(atoms))
            rows.append({"frame": ts.frame, "time_ps": ts.time, "n_contacts": n_contacts // 2})
        return pd.DataFrame(rows)

    def plot(self, result: pd.DataFrame) -> Figure:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(result["time_ps"] / 1000, result["n_contacts"])
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("# Cα contacts")
        ax.set_title("Residue contact count over time")
        fig.tight_layout()
        return fig
