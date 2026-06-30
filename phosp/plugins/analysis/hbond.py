from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
from MDAnalysis.exceptions import NoDataError
from matplotlib.figure import Figure
from phosp.plugins.analysis.base import AnalysisPlugin


class HBondPlugin(AnalysisPlugin):
    name = "hbond"

    def run(self, universe: mda.Universe, config: dict) -> pd.DataFrame:
        try:
            hba = HydrogenBondAnalysis(universe, between=["protein", "protein"])
            hba.run()
            data = hba.results.hbonds
            if len(data) == 0:
                return pd.DataFrame(columns=["frame", "donor_idx", "h_idx", "acceptor_idx", "distance", "angle"])
            return pd.DataFrame(data, columns=["frame", "donor_idx", "h_idx", "acceptor_idx", "distance", "angle"])
        except NoDataError:
            return pd.DataFrame(columns=["frame", "donor_idx", "h_idx", "acceptor_idx", "distance", "angle"])

    def plot(self, result: pd.DataFrame) -> Figure:
        fig, ax = plt.subplots(figsize=(8, 4))
        if not result.empty:
            counts = result.groupby("frame").size()
            ax.plot(counts.index, counts.values)
        ax.set_xlabel("Frame")
        ax.set_ylabel("# H-bonds")
        ax.set_title("Intra-protein H-bonds per Frame")
        fig.tight_layout()
        return fig
