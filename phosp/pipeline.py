from __future__ import annotations
import logging
from pathlib import Path

from phosp.config import PhospConfig
from phosp.exceptions import PhospError
from phosp.utils.checkpoint import Checkpoint

logger = logging.getLogger(__name__)

_ALL_STAGES = ["stage1", "stage2", "stage3", "stage4"]


class Pipeline:
    def __init__(self, config: PhospConfig, output_root: Path) -> None:
        self.config = config
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint = Checkpoint(output_root / "checkpoint.json")

    def execute(
        self,
        start_from: str | None = None,
        only_stages: str | None = None,
    ) -> None:
        stages = self._resolve_stages(start_from, only_stages)
        for stage_name in stages:
            if self.checkpoint.is_complete(stage_name):
                logger.info("Skipping %s (already complete)", stage_name)
                continue
            self._run_stage(stage_name)

    def _resolve_stages(self, start_from: str | None, only_stages: str | None) -> list[str]:
        if only_stages:
            nums = [s.strip() for s in only_stages.split(",")]
            stages = [f"stage{n}" for n in nums]
        else:
            stages = list(_ALL_STAGES)

        if start_from:
            try:
                idx = stages.index(start_from)
                stages = stages[idx:]
            except ValueError:
                raise PhospError(f"Unknown stage: {start_from}")
        return stages

    def _run_stage(self, stage_name: str) -> None:
        from phosp.engines.gromacs import GROMACSEngine
        from phosp.forcefields.charmm36m import CHARMM36mFF
        from phosp.forcefields.amber_ff14sb import AMBERff14SBFF

        engine = GROMACSEngine()
        ff = CHARMM36mFF() if self.config.forcefield == "charmm36m" else AMBERff14SBFF()
        output_dir = self.output_root / stage_name

        stage = self._build_stage(stage_name, engine, ff, output_dir)
        stage.validate_inputs()
        result = stage.run()
        self.checkpoint.mark_complete(stage_name, {k: str(v) for k, v in result.artifacts.items()})
        logger.info("Completed %s → %s", stage_name, result.output_dir)

    def _build_stage(self, stage_name: str, engine, ff, output_dir: Path):
        match stage_name:
            case "stage1":
                from phosp.stages.stage1_modify import Stage1Modify
                return Stage1Modify(self.config, engine, ff, output_dir)
            case "stage2":
                from phosp.stages.stage2_prepare import Stage2Prepare
                return Stage2Prepare(self.config, engine, ff, output_dir)
            case "stage3":
                from phosp.stages.stage3_simulate import Stage3Simulate
                return Stage3Simulate(self.config, engine, ff, output_dir)
            case "stage4":
                from phosp.stages.stage4_analyze import Stage4Analyze
                return Stage4Analyze(self.config, engine, ff, output_dir)
            case _:
                raise PhospError(f"Unknown stage: {stage_name}")
