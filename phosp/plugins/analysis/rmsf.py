from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import align, rms
from matplotlib.figure import Figure
from phosp.plugins.analysis.base import AnalysisPlugin


class RMSFPlugin(AnalysisPlugin):
    name = "rmsf"

    def run(self, universe: mda.Universe, config: dict) -> pd.DataFrame:
        selection = config.get("selection", "name CA")

        # RMSF must be computed on a trajectory aligned to its own average
        # structure — otherwise whole-body rotation/translation is mixed in
        # with internal flexibility, inflating values several-fold. Align a
        # fresh copy (not the passed-in universe, which stage4_analyze.py
        # shares across every other plugin) so this doesn't mutate coordinates
        # out from under plugins that run afterward.
        work_universe = mda.Universe(universe.filename, universe.trajectory.filename)
        average = align.AverageStructure(work_universe, work_universe, select=selection, ref_frame=0).run()
        align.AlignTraj(work_universe, average.results.universe, select=selection, in_memory=True).run()

        atoms = work_universe.select_atoms(selection)
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
