from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
import MDAnalysis as mda
from matplotlib.figure import Figure

# MDAnalysis's "protein" selection macro only recognizes standard residue
# names, so it silently excludes phospho-residues (SEP/TPO/PTR) — any plugin
# whose selection is meant to cover the whole polypeptide (not just
# unmodified residues) must use this instead of the bare "protein" macro.
PROTEIN_SELECTION = "(protein or resname SEP TPO PTR)"


class AnalysisPlugin(ABC):
    name: str

    @abstractmethod
    def run(self, universe: mda.Universe, config: dict) -> pd.DataFrame: ...

    @abstractmethod
    def plot(self, result: pd.DataFrame) -> Figure: ...
