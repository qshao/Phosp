from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import MDAnalysis as mda
from matplotlib.figure import Figure
from phosp.plugins.analysis.base import AnalysisPlugin


class DCCMPlugin(AnalysisPlugin):
    name = "dccm"

    def run(self, universe: mda.Universe, config: dict) -> pd.DataFrame:
        selection = config.get("selection", "name CA")
        ca = universe.select_atoms(selection)
        n_atoms = len(ca)
        positions = np.array([
            universe.select_atoms(selection).positions.copy()
            for ts in universe.trajectory
        ])

        mean_pos = positions.mean(axis=0)
        delta = positions - mean_pos
        n_frames = delta.shape[0]
        delta_flat = delta.reshape(n_frames, n_atoms * 3)

        cov = np.cov(delta_flat.T)
        dccm = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(n_atoms):
                c_ij = np.trace(cov[3*i:3*i+3, 3*j:3*j+3])
                c_ii = np.trace(cov[3*i:3*i+3, 3*i:3*i+3])
                c_jj = np.trace(cov[3*j:3*j+3, 3*j:3*j+3])
                denom = np.sqrt(c_ii * c_jj)
                dccm[i, j] = c_ij / denom if denom > 1e-10 else 0.0

        resids = ca.resids
        rows = [
            {"resid_i": int(resids[i]), "resid_j": int(resids[j]), "dcc": float(dccm[i, j])}
            for i in range(n_atoms) for j in range(n_atoms)
        ]
        return pd.DataFrame(rows)

    def plot(self, result: pd.DataFrame) -> Figure:
        matrix = result.pivot(index="resid_i", columns="resid_j", values="dcc").values
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r", origin="lower")
        plt.colorbar(im, ax=ax, label="DCC")
        ax.set_xlabel("Residue ID")
        ax.set_ylabel("Residue ID")
        ax.set_title("Dynamic Cross-Correlation Matrix")
        fig.tight_layout()
        return fig
