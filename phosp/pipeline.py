from __future__ import annotations
import hashlib
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
    def __init__(
        self,
        config: PhospConfig,
        output_root: Path,
        ui: PhospUI | None = None,
        config_path: Path | None = None,
        reference_mode: bool = False,
    ) -> None:
        self.config = config
        # GROMACS stage commands run with cwd=<stage dir> while also passing
        # <stage dir>-relative path arguments; a relative output_root would
        # make the subprocess re-resolve those arguments against its new cwd
        # and double-nest the path. Resolve once, here, so every derived
        # stage directory downstream is already absolute.
        self.output_root = Path(output_root).resolve()
        self.ui = ui
        self.reference_mode = reference_mode
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint = Checkpoint(output_root / "checkpoint.json")
        self._config_hash: str | None = None
        if config_path is not None:
            try:
                self._config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()[:16]
            except OSError as exc:
                logger.warning(
                    "Could not hash config file %s (%s) — the config-change "
                    "warning on resume will be unavailable for this run.",
                    config_path, exc,
                )

    def execute(
        self,
        start_from: str | None = None,
        only_stages: str | None = None,
        dry_run: bool = False,
    ) -> None:
        # Validate --stages/--start-from before preflight checks, so a typo
        # surfaces immediately instead of after a (possibly slow) GROMACS/
        # force-field/disk-space check.
        stages = self._resolve_stages(start_from, only_stages)
        if dry_run:
            logger.info("Dry run: would execute stages: %s", ", ".join(stages))
            return
        self._preflight_checks()
        if self._config_hash:
            stored = self.checkpoint.get_config_hash()
            if stored and stored != self._config_hash:
                logger.warning(
                    "Config has changed since this run was started. "
                    "Completed stages used the previous config. "
                    "Use --start-from stage1 to re-run from scratch."
                )
        self._clean_orphan_tmpdirs()
        for stage_name in stages:
            if self.checkpoint.is_complete(stage_name):
                # checkpoint.json only records that the stage succeeded once —
                # it never re-verifies the artifact directory is still on disk.
                # If it was deleted externally (cleanup, failed rsync, manual
                # edit) after completion, skipping it here would surface as a
                # confusing "run stageN first" error from whichever stage
                # runs next, with no hint the checkpoint itself is stale.
                if not (self.output_root / stage_name).exists():
                    logger.warning(
                        "checkpoint.json marks %s complete, but %s is missing "
                        "on disk — re-running it.",
                        stage_name, self.output_root / stage_name,
                    )
                    self._run_stage(stage_name)
                    continue
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
        from phosp.forcefields.discovery import discover_top_dir

        top_dir = discover_top_dir(self.config.gromacs.binary, "charmm36m-jul2022.ff")
        if top_dir is None:
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

        # Only the residue types this run's modification.sites/ncaa_sites
        # actually patch need to be registered as "Protein" in
        # residuetypes.dat, not a fixed phospho-only list — otherwise adding
        # a new PTM/ncAA type silently skips this check.
        from phosp.modification.base import get_modifier

        required = {
            get_modifier(site.mod_type, "charmm36m").new_resname
            for site in self.config.modification.sites
        }
        required |= {site.new_resname for site in self.config.modification.ncaa_sites}

        res_file = top_dir / "residuetypes.dat"
        content = res_file.read_text()
        missing = sorted(r for r in required if r not in content)
        if missing:
            printf_body = "".join(f"{r}\\tProtein\\n" for r in missing)
            raise PhospError(
                f"Residue types {missing} (needed for this run's modification.sites) "
                f"missing from residuetypes.dat.\n"
                f"Add them with:\n"
                f"  printf '{printf_body}' >> {res_file}"
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

        engine = GROMACSEngine(
            binary=self.config.gromacs.binary,
            timeout_minutes=self.config.gromacs.timeout_minutes,
        )
        ff = CHARMM36mFF() if self.config.forcefield == "charmm36m" else AMBERff14SBFF()
        stage = self._build_stage(stage_name, engine, ff, tmp_dir)

        if self.ui:
            self.ui.stage_start(stage_name)

        start = time.monotonic()
        try:
            self.checkpoint.mark_stage_started(stage_name)
            stage.validate_inputs()
            result = stage.run()
        except Exception as exc:
            if self.ui:
                self.ui.stage_error(stage_name, exc)
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            raise

        if final_dir.exists():
            shutil.rmtree(final_dir)
        tmp_dir.rename(final_dir)
        # The stage's real output (potentially a multi-hour GROMACS run) is now
        # safely on disk at final_dir. Bookkeeping below must never delete it on
        # failure — a bug here should surface as an unhandled exception with the
        # completed work intact, not silently destroy it while checkpoint.json
        # may already say the stage succeeded.
        remapped = self._remap_artifacts(result.artifacts, tmp_dir, final_dir)
        self.checkpoint.mark_complete(stage_name, remapped, config_hash=self._config_hash)
        elapsed = time.monotonic() - start
        if self.ui:
            self.ui.stage_complete(stage_name, elapsed)
        logger.info("Completed %s → %s", stage_name, final_dir)

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
                return Stage1Modify(self.config, engine, ff, output_dir, ui=self.ui, reference_mode=self.reference_mode)
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
