from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import MDAnalysis as mda
from matplotlib.figure import Figure
from phosp.plugins.analysis.base import AnalysisPlugin


class SASAPlugin(AnalysisPlugin):
    name = "sasa"

    def run(self, universe: mda.Universe, config: dict) -> pd.DataFrame:
        import freesasa
        target_resids = config.get("residues", [])

        rows = []
        classifier = freesasa.Classifier()

        for ts in universe.trajectory:
            if target_resids:
                sel = f"protein and resid {' '.join(str(r) for r in target_resids)}"
            else:
                sel = "protein"
            atoms = universe.select_atoms(sel)

            coords = atoms.positions.flatten().tolist()
            radii = []
            for atom in atoms:
                try:
                    r = classifier.radius(atom.resname, atom.name)
                except Exception:
                    r = 1.5
                radii.append(r)

            result = freesasa.calcCoord(coords, radii)
            rows.append({
                "frame": ts.frame,
                "time_ps": ts.time,
                "sasa_angstrom2": result.totalArea(),
            })
        return pd.DataFrame(rows)

    def plot(self, result: pd.DataFrame) -> Figure:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(result["time_ps"] / 1000, result["sasa_angstrom2"])
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("SASA (Å²)")
        ax.set_title("Solvent Accessible Surface Area")
        fig.tight_layout()
        return fig
