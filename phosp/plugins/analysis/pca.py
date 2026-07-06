from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import pca as mda_pca
from matplotlib.figure import Figure
from phosp.plugins.analysis.base import AnalysisPlugin


class PCAPlugin(AnalysisPlugin):
    name = "pca"

    def run(self, universe: mda.Universe, config: dict) -> pd.DataFrame:
        selection = config.get("selection", "name CA")

        # Without alignment, PC1/PC2 are dominated by whole-body rotation/
        # translation rather than internal conformational motion — the same
        # class of bug already fixed in rmsf.py. Align on an independent copy
        # (not the passed-in universe, which stage4_analyze.py shares across
        # every other plugin) since MDAnalysis's align=True mutates atom
        # positions in place via _fit_to().
        work_universe = mda.Universe(universe.filename, universe.trajectory.filename)
        pc = mda_pca.PCA(work_universe, select=selection, align=True).run()
        atoms = work_universe.select_atoms(selection)
        projected = pc.transform(atoms, n_components=2)
        df = pd.DataFrame({
            "frame": range(len(projected)),
            "pc1": projected[:, 0],
            "pc2": projected[:, 1],
        })
        df.attrs["explained_variance_ratio"] = pc.results.variance[:2].tolist()
        return df

    def plot(self, result: pd.DataFrame) -> Figure:
        ev = result.attrs.get("explained_variance_ratio", [0, 0])
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(result["pc1"], result["pc2"],
                        c=result["frame"], cmap="viridis", s=10)
        plt.colorbar(sc, ax=ax, label="Frame")
        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)" if ev[0] else "PC1")
        ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)" if ev[1] else "PC2")
        ax.set_title("PCA Projection (Cα)")
        fig.tight_layout()
        return fig
