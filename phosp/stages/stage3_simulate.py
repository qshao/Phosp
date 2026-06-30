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
        for phase in _PHASES:
            mdp = stage2_dir / f"{phase}.mdp"
            if not mdp.exists():
                raise StageInputError(
                    f"{phase}.mdp not found in {stage2_dir}. Run stage2 first."
                )

    def run(self) -> StageResult:
        out = self.output_root
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
            phase_dir.mkdir(parents=True, exist_ok=True)
            mdp = stage2_dir / f"{phase}.mdp"
            restraint = structure if phase in _RESTRAINT_PHASES else None
            result = self.engine.run_phase(
                phase=phase,
                mdp=mdp,
                topology=topology,
                structure=current_structure,
                output_dir=phase_dir,
                restraint_gro=restraint,
                gpu_id=self.config.simulation.gpu_id,
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
        # final_out: the path stage3 will have after the tmp→final rename.
        # The script file is written to out (tmp dir, which exists), while the
        # paths inside the script reference final_out (the location the HPC job
        # will see at submission time).
        final_out = out.parent / out.name.lstrip('.').removesuffix('_tmp') if out.name.startswith('.') else out
        for phase in _PHASES:
            (out / phase).mkdir(parents=True, exist_ok=True)
        script = self.engine.generate_hpc_script(
            scheduler=hpc.scheduler,
            resources=resources,
            phases=_PHASES,
            output_dir=out,       # write here (tmp dir — exists now, gets renamed)
            work_dir=final_out,   # WORK= path referenced inside the script
        )
        if hpc.auto_submit:
            cmd = "sbatch" if hpc.scheduler == "slurm" else "qsub"
            subprocess.run([cmd, str(script)], check=True)
            logger.info("Submitted HPC job: %s", script)
        else:
            logger.info("HPC script written (not submitted): %s", script)
        return StageResult(stage="stage3", output_dir=out, artifacts={"hpc_script": script})
