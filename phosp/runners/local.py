from __future__ import annotations
import logging
from pathlib import Path

from phosp.runners.base import SimulationRunner

logger = logging.getLogger(__name__)


class LocalRunner(SimulationRunner):
    """Run all MD phases directly on the local machine."""

    def run(self, phases, restraint_phases, topology, structure, mdp_dir,
            output_dir, work_dir, engine, gpu_id):
        current_structure = structure
        artifacts: dict[str, Path] = {}
        for phase in phases:
            phase_dir = output_dir / phase
            phase_dir.mkdir(parents=True, exist_ok=True)
            restraint = structure if phase in restraint_phases else None
            engine.run_phase(
                phase=phase,
                mdp=mdp_dir / f"{phase}.mdp",
                topology=topology,
                structure=current_structure,
                output_dir=phase_dir,
                restraint_gro=restraint,
                gpu_id=gpu_id,
            )
            next_gro = phase_dir / f"{phase}.gro"
            if next_gro.exists():
                current_structure = next_gro
            artifacts[phase] = phase_dir
            logger.info("Completed phase: %s", phase)
        return artifacts
