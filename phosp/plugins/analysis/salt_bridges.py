from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance
from matplotlib.figure import Figure
from phosp.plugins.analysis.base import AnalysisPlugin


class SaltBridgesPlugin(AnalysisPlugin):
    name = "salt_bridges"

    def run(self, universe: mda.Universe, config: dict) -> pd.DataFrame:
        cutoff = config.get("cutoff_angstrom", 4.0)
        acidic = universe.select_atoms("(resname ASP GLU) and (name OD1 OD2 OE1 OE2)")
        basic = universe.select_atoms(
            "(resname LYS and name NZ) or (resname ARG and name NH1 NH2 NE)"
        )
        rows = []
        for ts in universe.trajectory:
            if len(acidic) == 0 or len(basic) == 0:
                continue
            pairs, dists = capped_distance(
                acidic.positions, basic.positions, max_cutoff=cutoff, return_distances=True
            )
            for (i, j), d in zip(pairs, dists):
                rows.append({
                    "frame": ts.frame,
                    "acidic_resid": int(acidic[i].resid),
                    "acidic_resname": acidic[i].resname,
                    "basic_resid": int(basic[j].resid),
                    "basic_resname": basic[j].resname,
                    "distance_angstrom": float(d),
                })
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["frame", "acidic_resid", "acidic_resname",
                     "basic_resid", "basic_resname", "distance_angstrom"]
        )

    def plot(self, result: pd.DataFrame) -> Figure:
        fig, ax = plt.subplots(figsize=(8, 4))
        if not result.empty:
            counts = result.groupby("frame").size()
            ax.plot(counts.index, counts.values)
        ax.set_xlabel("Frame")
        ax.set_ylabel("# Salt bridges")
        ax.set_title(f"Salt bridges per frame (cutoff ≤ {result.get('distance_angstrom', pd.Series([4.0])).max():.1f} Å)")
        fig.tight_layout()
        return fig
