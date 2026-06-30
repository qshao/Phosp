from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path


class SimulationRunner(ABC):
    @abstractmethod
    def run(
        self,
        phases: list[str],
        restraint_phases: set[str],
        topology: Path,
        structure: Path,
        mdp_dir: Path,
        output_dir: Path,
        work_dir: Path,
        engine,
        gpu_id: int | None,
    ) -> dict[str, Path]:
        """Execute all MD phases and return {phase_name: phase_output_dir}."""
