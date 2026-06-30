from __future__ import annotations
import logging
import subprocess
from pathlib import Path

from phosp.exceptions import StageInputError
from phosp.stages.base import Stage, StageResult

logger = logging.getLogger(__name__)

_PHASES = ["minimization", "nvt", "npt", "production"]
_RESTRAINT_PHASES = {"nvt", "npt"}


class Stage3Simulate(Stage):
    def validate_inputs(self) -> None:
        stage2_dir = self.output_root.parent / "stage2"
        if not (stage2_dir / "ions.gro").exists():
            raise StageInputError(
                f"ions.gro not found in {stage2_dir}. Run stage2 first."
            )

    def run(self) -> StageResult:
        out = self.output_root
        out.mkdir(parents=True, exist_ok=True)
        cfg = self.config
        stage2_dir = out.parent / "stage2"
        topology = stage2_dir / "topol.top"
        structure = stage2_dir / "ions.gro"
        hpc = cfg.simulation.hpc

        if hpc.enabled:
            return self._hpc_run(out, topology, structure, stage2_dir, hpc)
        return self._direct_run(out, topology, structure, stage2_dir)

    def _direct_run(self, out, topology, structure, stage2_dir) -> StageResult:
        current_structure = structure
        artifacts: dict[str, Path] = {}
        for phase in _PHASES:
            phase_dir = out / phase
            phase_dir.mkdir(exist_ok=True)
            mdp = stage2_dir / f"{phase}.mdp"
            restraint = structure if phase in _RESTRAINT_PHASES else None
            result = self.engine.run_phase(
                phase=phase,
                mdp=mdp,
                topology=topology,
                structure=current_structure,
                output_dir=phase_dir,
                restraint_gro=restraint,
            )
            next_gro = phase_dir / f"{phase}.gro"
            if next_gro.exists():
                current_structure = next_gro
            artifacts[phase] = phase_dir
            logger.info("Completed phase: %s", phase)

        return StageResult(stage="stage3", output_dir=out, artifacts=artifacts)

    def _hpc_run(self, out, topology, structure, stage2_dir, hpc) -> StageResult:
        resources = {
            "ntasks": hpc.ntasks,
            "gpus": hpc.gpus,
            "walltime": hpc.walltime,
            "partition": hpc.partition,
        }
        for phase in _PHASES:
            (out / phase).mkdir(exist_ok=True)
        script = self.engine.generate_hpc_script(
            scheduler=hpc.scheduler,
            resources=resources,
            phases=_PHASES,
            output_dir=out,
        )
        if hpc.auto_submit:
            cmd = "sbatch" if hpc.scheduler == "slurm" else "qsub"
            subprocess.run([cmd, str(script)], check=True)
            logger.info("Submitted HPC job: %s", script)
        else:
            logger.info("HPC script written (not submitted): %s", script)
        return StageResult(stage="stage3", output_dir=out, artifacts={"hpc_script": script})
