from __future__ import annotations
import logging
from pathlib import Path

from phosp.exceptions import StageInputError
from phosp.runners.factory import get_runner
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
        sim = self.config.simulation
        stage2_dir = out.parent / "stage2"
        topology = stage2_dir / "topol.top"
        structure = stage2_dir / "ions.gro"

        # final_out: the path stage3 will have after the tmp→final rename;
        # HPC runners embed this in job scripts so paths stay correct post-rename.
        final_out = (
            out.parent / out.name.lstrip(".").removesuffix("_tmp")
            if out.name.startswith(".")
            else out
        )

        runner = get_runner(sim)
        artifacts = runner.run(
            phases=_PHASES,
            restraint_phases=_RESTRAINT_PHASES,
            topology=topology,
            structure=structure,
            mdp_dir=stage2_dir,
            output_dir=out,
            work_dir=final_out,
            engine=self.engine,
            gpu_id=sim.gpu_id,
        )
        return StageResult(stage="stage3", output_dir=out, artifacts=artifacts)
