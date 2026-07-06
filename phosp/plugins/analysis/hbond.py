from __future__ import annotations
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
from MDAnalysis.exceptions import NoDataError
from matplotlib.figure import Figure
from phosp.plugins.analysis.base import AnalysisPlugin, PROTEIN_SELECTION

logger = logging.getLogger(__name__)
_COLUMNS = ["frame", "donor_idx", "h_idx", "acceptor_idx", "distance", "angle"]


class HBondPlugin(AnalysisPlugin):
    name = "hbond"

    def run(self, universe: mda.Universe, config: dict) -> pd.DataFrame:
        try:
            hba = HydrogenBondAnalysis(universe, between=[PROTEIN_SELECTION, PROTEIN_SELECTION])
            hba.run()
            data = hba.results.hbonds
            if len(data) == 0:
                logger.warning("hbond plugin found zero hydrogen bonds — check the selection/topology if this is unexpected")
                return pd.DataFrame(columns=_COLUMNS)
            return pd.DataFrame(data, columns=_COLUMNS)
        except NoDataError as exc:
            logger.warning("hbond plugin: HydrogenBondAnalysis raised NoDataError (%s); returning empty result", exc)
            return pd.DataFrame(columns=_COLUMNS)

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
