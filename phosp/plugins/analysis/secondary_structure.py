from __future__ import annotations
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import MDAnalysis as mda
from matplotlib.figure import Figure
from phosp.plugins.analysis.base import AnalysisPlugin, PROTEIN_SELECTION


class SecondaryStructurePlugin(AnalysisPlugin):
    name = "secondary_structure"

    def run(self, universe: mda.Universe, config: dict) -> pd.DataFrame:
        try:
            from MDAnalysis.analysis.dssp import DSSP
            protein = universe.select_atoms(PROTEIN_SELECTION)
            dssp = DSSP(protein).run()
            resids = protein.residues.resids
            rows = []
            for i, ts in enumerate(universe.trajectory):
                for j, resid in enumerate(resids):
                    rows.append({
                        "frame": ts.frame,
                        "time_ps": ts.time,
                        "resid": resid,
                        "ss": dssp.results.dssp[i, j],
                    })
            return pd.DataFrame(rows)
        except ImportError:
            logging.getLogger(__name__).warning(
                "MDAnalysis DSSP not available; secondary_structure plugin returns empty DataFrame"
            )
            return pd.DataFrame(columns=["frame", "time_ps", "resid", "ss"])

    def plot(self, result: pd.DataFrame) -> Figure:
        fig, ax = plt.subplots(figsize=(10, 5))
        if result.empty:
            ax.text(0.5, 0.5, "DSSP data unavailable", ha="center", va="center")
            return fig
        pivot = result.pivot_table(index="resid", columns="frame", values="ss", aggfunc="first")
        ss_codes = {"H": 1, "E": 2, "C": 0, "B": 3, "T": 4, "S": 5, "G": 6, "I": 7}
        numeric = pivot.map(lambda x: ss_codes.get(x, 0))
        ax.imshow(numeric.values, aspect="auto", cmap="tab10", interpolation="nearest")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Residue ID")
        ax.set_title("Secondary Structure Evolution")
        fig.tight_layout()
        return fig
