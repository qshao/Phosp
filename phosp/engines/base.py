from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SimulationResult:
    phase: str
    output_dir: Path
    success: bool
    log_path: Path


class MDEngine(ABC):
    @abstractmethod
    def prepare_topology(self, pdb: Path, forcefield: object) -> Path: ...

    @abstractmethod
    def solvate(self, gro: Path, topology: Path, box_type: str, water_model: str) -> tuple[Path, Path]: ...

    @abstractmethod
    def add_ions(self, gro: Path, topology: Path, concentration_mM: float, neutralize: bool) -> tuple[Path, Path]: ...

    @abstractmethod
    def generate_mdp(self, phase: str, protocol: dict, output_dir: Path) -> Path: ...

    @abstractmethod
    def run_phase(self, phase: str, mdp: Path, topology: Path, structure: Path, output_dir: Path, restraint_gro: Path | None = None) -> SimulationResult: ...

    @abstractmethod
    def generate_hpc_script(self, scheduler: str, resources: dict, phases: list[str], output_dir: Path) -> Path: ...
