from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
import MDAnalysis as mda
from matplotlib.figure import Figure


class AnalysisPlugin(ABC):
    name: str

    @abstractmethod
    def run(self, universe: mda.Universe, config: dict) -> pd.DataFrame: ...

    @abstractmethod
    def plot(self, result: pd.DataFrame) -> Figure: ...
