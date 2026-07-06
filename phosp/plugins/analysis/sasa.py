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
        target_resids = set(config.get("residues", []))

        # MDAnalysis's "protein" selection macro only recognizes standard
        # residue names, so it silently excludes phospho-residues (SEP/TPO/PTR)
        # — include them explicitly or a phosphosite selection returns 0 atoms.
        protein_sel = "(protein or resname SEP TPO PTR)"
        # Per-residue SASA must be computed with the *entire* protein present
        # so neighboring residues can sterically shield each other — running
        # freesasa on just the target residues in isolation reports the SASA
        # of a disconnected fragment, not the true in-context value.
        atoms = universe.select_atoms(protein_sel)
        target_idx = (
            [i for i, atom in enumerate(atoms) if atom.resid in target_resids]
            if target_resids else None
        )

        classifier = freesasa.Classifier()
        radii = []
        for atom in atoms:
            try:
                r = classifier.radius(atom.resname, atom.name)
            except Exception:
                r = -1.0
            # freesasa returns -1.0 (no exception) for atoms it doesn't
            # recognize — e.g. CHARMM atom names on phospho-residues
            # (SEP/TPO/PTR) — which corrupts the area calculation.
            if r < 0:
                r = 1.5
            radii.append(r)

        rows = []
        for ts in universe.trajectory:
            coords = atoms.positions.flatten().tolist()
            result = freesasa.calcCoord(coords, radii)
            row = {
                "frame": ts.frame,
                "time_ps": ts.time,
                "sasa_angstrom2": result.totalArea(),
            }
            if target_idx is not None:
                # residueAreas()'s dict-based breakdown silently drops
                # non-standard residue names (e.g. TPO), so sum per-atom
                # areas by index instead — robust regardless of naming.
                row["sasa_target_angstrom2"] = sum(result.atomArea(i) for i in target_idx)
            rows.append(row)
        return pd.DataFrame(rows)

    def plot(self, result: pd.DataFrame) -> Figure:
        has_target = "sasa_target_angstrom2" in result.columns
        column = "sasa_target_angstrom2" if has_target else "sasa_angstrom2"
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(result["time_ps"] / 1000, result[column])
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("SASA (Å²)")
        ax.set_title("Solvent Accessible Surface Area" + (" (selected residues)" if has_target else ""))
        fig.tight_layout()
        return fig
