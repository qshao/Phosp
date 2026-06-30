from __future__ import annotations
import logging
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

from phosp.config import PhospConfig
from phosp.engines.gromacs import GROMACSEngine
from phosp.exceptions import PhospError
from phosp.utils.checkpoint import Checkpoint

if TYPE_CHECKING:
    from phosp.ui import PhospUI

logger = logging.getLogger(__name__)

_ALL_STAGES = ["stage1", "stage2", "stage3", "stage4"]
_DISK_WARN_GB = 10.0


class Pipeline:
    def __init__(self, config: PhospConfig, output_root: Path, ui: PhospUI | None = None) -> None:
        self.config = config
        self.output_root = output_root
        self.ui = ui
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint = Checkpoint(output_root / "checkpoint.json")

    def execute(
        self,
        start_from: str | None = None,
        only_stages: str | None = None,
        dry_run: bool = False,
    ) -> None:
        if dry_run:
            stages = self._resolve_stages(start_from, only_stages)
            logger.info("Dry run: would execute stages: %s", ", ".join(stages))
            return
        self._preflight_checks()
        self._clean_orphan_tmpdirs()
        stages = self._resolve_stages(start_from, only_stages)
        for stage_name in stages:
            if self.checkpoint.is_complete(stage_name):
                logger.info("Skipping %s (already complete)", stage_name)
                continue
            self._run_stage(stage_name)

    def _preflight_checks(self) -> None:
        binary = self.config.gromacs.binary
        if shutil.which(binary) is None and not Path(binary).is_file():
            raise PhospError(
                f"GROMACS binary '{binary}' not found. "
                "Set gromacs.binary in your config to the correct path or binary name."
            )
        self._check_forcefield()
        self._warn_disk_space()

    def _check_forcefield(self) -> None:
        ff = self.config.forcefield
        if ff == "amber_ff14sb":
            raise PhospError(
                "AMBER ff14SB is not yet fully supported with GROMACS.\n"
                "Use forcefield: charmm36m instead.\n"
                "(AMBER ff14SB support is planned for a future release.)"
            )
        if ff != "charmm36m":
            return
        import re
        import subprocess
        try:
            out = subprocess.run(
                [self.config.gromacs.binary, "--version"], capture_output=True, text=True
            ).stdout
            m = re.search(r"Data prefix:\s*(.+)", out)
            if not m:
                return
            top_dir = Path(m.group(1).strip()) / "share" / "gromacs" / "top"
        except Exception:
            return

        ff_dir = top_dir / "charmm36m-jul2022.ff"
        if not ff_dir.exists():
            raise PhospError(
                "CHARMM36m force field not found at:\n"
                f"  {ff_dir}\n\n"
                "Install it with:\n"
                "  curl -O https://mackerell.umaryland.edu/download.php"
                "?filename=CHARMM_ff_params_files/charmm36-jul2022.ff.tgz\n"
                f"  tar -xzf charmm36-jul2022.ff.tgz -C {top_dir}\n"
                f"  ln -s {top_dir}/charmm36-jul2022.ff {ff_dir}\n\n"
                "See README.md for the full setup procedure."
            )

        res_file = top_dir / "residuetypes.dat"
        missing = [
            r for r in ("SEP", "TPO", "PTR")
            if r not in res_file.read_text()
        ]
        if missing:
            raise PhospError(
                f"Phospho-residue types {missing} missing from residuetypes.dat.\n"
                f"Add them with:\n"
                f"  printf 'SEP\\tProtein\\nTPO\\tProtein\\nPTR\\tProtein\\n'"
                f" >> {res_file}"
            )

    def _warn_disk_space(self) -> None:
        try:
            available_gb = shutil.disk_usage(self.output_root).free / (1024 ** 3)
            if available_gb < _DISK_WARN_GB:
                logger.warning(
                    "Low disk space: minimum %.1f GB needed, %.1f GB available at %s",
                    _DISK_WARN_GB, available_gb, self.output_root,
                )
        except OSError:
            pass

    def _clean_orphan_tmpdirs(self) -> None:
        for d in self.output_root.glob(".stage*_tmp"):
            if d.is_dir():
                logger.warning("Removing orphaned temp dir: %s", d)
                shutil.rmtree(d)

    def _resolve_stages(self, start_from: str | None, only_stages: str | None) -> list[str]:
        if only_stages:
            stages = [f"stage{n.strip()}" for n in only_stages.split(",")]
        else:
            stages = list(_ALL_STAGES)

        invalid = [s for s in stages if s not in _ALL_STAGES]
        if invalid:
            raise PhospError(
                f"Unknown stage(s): {invalid}. Valid stages: {_ALL_STAGES}"
            )

        if start_from:
            if start_from not in _ALL_STAGES:
                raise PhospError(
                    f"Unknown stage: {start_from!r}. Valid stages: {_ALL_STAGES}"
                )
            try:
                stages = stages[stages.index(start_from):]
            except ValueError:
                raise PhospError(
                    f"--start-from={start_from!r} is not in the stages to run: {stages}"
                )
        return stages

    def _run_stage(self, stage_name: str) -> None:
        from phosp.forcefields.charmm36m import CHARMM36mFF
        from phosp.forcefields.amber_ff14sb import AMBERff14SBFF

        final_dir = self.output_root / stage_name
        tmp_dir = self.output_root / f".{stage_name}_tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        engine = GROMACSEngine(binary=self.config.gromacs.binary)
        ff = CHARMM36mFF() if self.config.forcefield == "charmm36m" else AMBERff14SBFF()
        stage = self._build_stage(stage_name, engine, ff, tmp_dir)

        if self.ui:
            self.ui.stage_start(stage_name)

        start = time.monotonic()
        renamed = False
        try:
            stage.validate_inputs()
            result = stage.run()
            if final_dir.exists():
                shutil.rmtree(final_dir)
            tmp_dir.rename(final_dir)
            renamed = True
            remapped = self._remap_artifacts(result.artifacts, tmp_dir, final_dir)
            self.checkpoint.mark_complete(stage_name, remapped)
            elapsed = time.monotonic() - start
            if self.ui:
                self.ui.stage_complete(stage_name, elapsed)
            logger.info("Completed %s → %s", stage_name, final_dir)
        except Exception as exc:
            if self.ui:
                self.ui.stage_error(stage_name, exc)
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            # Only delete final_dir if we put it there in this run; otherwise a
            # prior successful run's output is still valid and should be kept.
            if renamed and final_dir.exists():
                shutil.rmtree(final_dir)
            raise

    @staticmethod
    def _remap_artifacts(artifacts: dict, from_dir: Path, to_dir: Path) -> dict[str, str]:
        result = {}
        for k, v in artifacts.items():
            try:
                result[k] = str(to_dir / Path(str(v)).relative_to(from_dir))
            except ValueError:
                result[k] = str(v)
        return result

    def _build_stage(self, stage_name: str, engine, ff, output_dir: Path):
        match stage_name:
            case "stage1":
                from phosp.stages.stage1_modify import Stage1Modify
                return Stage1Modify(self.config, engine, ff, output_dir, ui=self.ui)
            case "stage2":
                from phosp.stages.stage2_prepare import Stage2Prepare
                return Stage2Prepare(self.config, engine, ff, output_dir, ui=self.ui)
            case "stage3":
                from phosp.stages.stage3_simulate import Stage3Simulate
                return Stage3Simulate(self.config, engine, ff, output_dir, ui=self.ui)
            case "stage4":
                from phosp.stages.stage4_analyze import Stage4Analyze
                return Stage4Analyze(self.config, engine, ff, output_dir, ui=self.ui)
            case _:
                raise PhospError(f"Unknown stage: {stage_name}")
