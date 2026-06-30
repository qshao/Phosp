# phosp UX & Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add atomic stage writes, Rich terminal progress, pre-flight guardrails, plugin fail-soft, and logging configuration to the phosp pipeline.

**Architecture:** Six independent tasks layered from bottom up — logging and UI utilities first (no dependencies), then config validators, then the pipeline core (atomic writes + guardrails), then CLI commands that wire everything together, and finally Stage 4 fail-soft. Each task is independently testable.

**Tech Stack:** Python ≥ 3.10, Pydantic v2, Typer, `rich` (already a transitive dep of Typer — no new install required), MDAnalysis, Jinja2.

## Global Constraints

- Python ≥ 3.10; `from __future__ import annotations` in every new file
- No `print()` in library code — use `logging.getLogger(__name__)`
- All file I/O via `pathlib.Path`
- Pydantic v2: `@field_validator`, `@model_validator(mode="after")`, `model_config`
- No comments unless WHY is non-obvious
- `rich` requires no new dependency — it is already installed via Typer

---

### Task 1: Logging Configuration

**Files:**
- Create: `phosp/logging.py`
- Create: `tests/test_logging.py`

**Interfaces:**
- Produces: `configure_logging(level: str = "INFO", log_file: Path | None = None) -> None` — attaches handlers to the `"phosp"` root logger; idempotent (won't double-add StreamHandler if called twice)

---

- [ ] **Step 1: Write failing tests**

```python
# tests/test_logging.py
from __future__ import annotations
import logging
from pathlib import Path
from phosp.logging import configure_logging


def _clear_phosp_handlers() -> None:
    logging.getLogger("phosp").handlers.clear()


def test_configure_logging_attaches_stream_handler():
    _clear_phosp_handlers()
    configure_logging("WARNING")
    root = logging.getLogger("phosp")
    assert root.level == logging.WARNING
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], logging.StreamHandler)
    _clear_phosp_handlers()


def test_configure_logging_idempotent():
    _clear_phosp_handlers()
    configure_logging()
    configure_logging()
    assert len(logging.getLogger("phosp").handlers) == 1
    _clear_phosp_handlers()


def test_configure_logging_adds_file_handler(tmp_path: Path):
    _clear_phosp_handlers()
    log_file = tmp_path / "phosp.log"
    configure_logging(log_file=log_file)
    root = logging.getLogger("phosp")
    assert len(root.handlers) == 2
    handler_types = {type(h) for h in root.handlers}
    assert logging.StreamHandler in handler_types
    assert logging.FileHandler in handler_types
    _clear_phosp_handlers()
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_logging.py -v
```
Expected: `ImportError: cannot import name 'configure_logging'`

- [ ] **Step 3: Implement**

```python
# phosp/logging.py
from __future__ import annotations
import logging
from pathlib import Path


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    root = logging.getLogger("phosp")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_logging.py -v
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add phosp/logging.py tests/test_logging.py
git commit -m "feat: configure_logging utility for phosp root logger"
```

---

### Task 2: PhospUI Class

**Files:**
- Create: `phosp/ui.py`
- Create: `tests/test_ui.py`

**Interfaces:**
- Produces: `PhospUI(console=None)` — `stage_start(name, description="")`, `stage_complete(name, elapsed_s)`, `stage_error(name, exc)`, `plugin_start(plugin_name)`
- `name` values: `"stage1"`, `"stage2"`, `"stage3"`, `"stage4"`

---

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ui.py
from __future__ import annotations
from io import StringIO
from rich.console import Console
from phosp.ui import PhospUI


def _ui() -> PhospUI:
    return PhospUI(console=Console(file=StringIO(), force_terminal=False))


def test_stage_lifecycle_no_error():
    ui = _ui()
    ui.stage_start("stage1")
    ui.stage_complete("stage1", 5.0)


def test_stage_error_no_crash():
    ui = _ui()
    ui.stage_start("stage2")
    ui.stage_error("stage2", ValueError("topology missing"))


def test_plugin_start_no_crash():
    ui = _ui()
    ui.stage_start("stage4")
    ui.plugin_start("rmsd")
    ui.plugin_start("rmsf")
    ui.stage_complete("stage4", 42.0)


def test_elapsed_formats_minutes():
    ui = _ui()
    ui.stage_start("stage3")
    ui.stage_complete("stage3", 125.0)  # 2m 5s — no assertion; just no crash


def test_unknown_stage_name_no_crash():
    ui = _ui()
    ui.stage_start("unknown_stage")
    ui.stage_complete("unknown_stage", 1.0)
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_ui.py -v
```
Expected: `ImportError: cannot import name 'PhospUI'`

- [ ] **Step 3: Implement**

```python
# phosp/ui.py
from __future__ import annotations
import time
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner

_STAGE_LABELS: dict[str, str] = {
    "stage1": "Stage 1 — Chemical Modification",
    "stage2": "Stage 2 — MD Preparation",
    "stage3": "Stage 3 — MD Simulation",
    "stage4": "Stage 4 — Analysis",
}


class PhospUI:
    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()
        self._live: Live | None = None
        self._start: float = 0.0

    def stage_start(self, name: str, description: str = "") -> None:
        label = _STAGE_LABELS.get(name, name)
        self._console.print(f"\n[bold cyan]▶  {label}[/]")
        self._start = time.monotonic()
        spinner = Spinner("dots", text=f"  {description or 'Running...'}")
        self._live = Live(spinner, console=self._console, refresh_per_second=10)
        self._live.start()

    def stage_complete(self, name: str, elapsed_s: float) -> None:
        self._stop_live()
        label = _STAGE_LABELS.get(name, name)
        mins, secs = divmod(int(elapsed_s), 60)
        elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"
        self._console.print(f"[green]✓  {label} complete[/]  ({elapsed_str})")

    def stage_error(self, name: str, exc: Exception) -> None:
        self._stop_live()
        label = _STAGE_LABELS.get(name, name)
        self._console.print(Panel(
            f"[bold]{type(exc).__name__}:[/] {exc}",
            title=f"[red]✗ {label} failed[/]",
            border_style="red",
        ))

    def plugin_start(self, plugin_name: str) -> None:
        if self._live:
            self._live.update(Spinner("dots", text=f"  Running plugin: {plugin_name}"))

    def _stop_live(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_ui.py -v
```
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add phosp/ui.py tests/test_ui.py
git commit -m "feat: PhospUI rich terminal progress display"
```

---

### Task 3: Config Validators

**Files:**
- Modify: `phosp/config.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Consumes: existing `SimulationConfig`, `HPCConfig` from `phosp/config.py`
- Produces: same classes, now rejecting invalid values at construction time

---

- [ ] **Step 1: Write failing tests**

Add to `tests/test_config.py`:

```python
import pytest
from pydantic import ValidationError
from phosp.config import SimulationConfig, HPCConfig


def test_production_time_must_be_positive():
    with pytest.raises(ValidationError, match="production_time_ns"):
        SimulationConfig(production_time_ns=0.0)


def test_production_time_negative_rejected():
    with pytest.raises(ValidationError, match="production_time_ns"):
        SimulationConfig(production_time_ns=-5.0)


def test_output_freq_must_be_positive():
    with pytest.raises(ValidationError, match="output_freq_ps"):
        SimulationConfig(output_freq_ps=0.0)


def test_output_freq_cannot_exceed_production():
    # production=1ns=1000ps, output_freq=2000ps > 1000ps → invalid
    with pytest.raises(ValidationError, match="output_freq_ps"):
        SimulationConfig(production_time_ns=1.0, output_freq_ps=2000.0)


def test_valid_output_freq_equal_to_production():
    # 100ns = 100000ps, output_freq=100000ps is exactly equal → valid
    cfg = SimulationConfig(production_time_ns=100.0, output_freq_ps=100000.0)
    assert cfg.output_freq_ps == 100000.0


def test_salt_concentration_non_negative():
    with pytest.raises(ValidationError, match="salt_concentration_mM"):
        SimulationConfig(salt_concentration_mM=-1.0)


def test_salt_concentration_zero_allowed():
    cfg = SimulationConfig(salt_concentration_mM=0.0)
    assert cfg.salt_concentration_mM == 0.0


def test_hpc_ntasks_must_be_at_least_1():
    with pytest.raises(ValidationError, match="ntasks"):
        HPCConfig(ntasks=0)


def test_hpc_gpus_non_negative():
    with pytest.raises(ValidationError, match="gpus"):
        HPCConfig(gpus=-1)
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_config.py -v
```
Expected: several `FAILED` — the validators don't exist yet.

- [ ] **Step 3: Add validators to `phosp/config.py`**

Add `field_validator` and `model_validator` imports at the top (they are already imported in the file as part of `from pydantic import BaseModel, Field, model_validator` — add `field_validator` to that import).

Replace the `HPCConfig` class:

```python
class HPCConfig(BaseModel):
    enabled: bool = False
    scheduler: Literal["slurm", "pbs"] = "slurm"
    ntasks: int = 8
    gpus: int = 1
    walltime: str = "24:00:00"
    partition: str = "gpu"
    auto_submit: bool = False

    @field_validator("ntasks")
    @classmethod
    def ntasks_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("ntasks must be >= 1")
        return v

    @field_validator("gpus")
    @classmethod
    def gpus_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("gpus must be >= 0")
        return v
```

Replace the `SimulationConfig` class:

```python
class SimulationConfig(BaseModel):
    production_time_ns: float = 100.0
    output_freq_ps: float = 10.0
    water_model: Literal["tip3p", "spce"] = "tip3p"
    box_type: Literal["dodecahedron", "cubic"] = "dodecahedron"
    salt_concentration_mM: float = 150.0
    hpc: HPCConfig = Field(default_factory=HPCConfig)

    @field_validator("production_time_ns")
    @classmethod
    def production_time_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("production_time_ns must be > 0")
        return v

    @field_validator("output_freq_ps")
    @classmethod
    def output_freq_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("output_freq_ps must be > 0")
        return v

    @field_validator("salt_concentration_mM")
    @classmethod
    def salt_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("salt_concentration_mM must be >= 0")
        return v

    @model_validator(mode="after")
    def output_freq_fits_in_production(self) -> SimulationConfig:
        if self.output_freq_ps > self.production_time_ns * 1000:
            raise ValueError(
                f"output_freq_ps ({self.output_freq_ps} ps) exceeds "
                f"production_time_ns ({self.production_time_ns} ns = "
                f"{self.production_time_ns * 1000} ps) — no frames would be written"
            )
        return self
```

Update the top-level import in `phosp/config.py` from:
```python
from pydantic import BaseModel, Field, model_validator
```
to:
```python
from pydantic import BaseModel, Field, field_validator, model_validator
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_config.py -v
```
Expected: all tests pass (existing + 9 new).

- [ ] **Step 5: Commit**

```bash
git add phosp/config.py tests/test_config.py
git commit -m "feat: Pydantic field validators for SimulationConfig and HPCConfig"
```

---

### Task 4: Atomic Stage Writes + Pipeline Guardrails

**Files:**
- Modify: `phosp/stages/base.py` — add `ui` param
- Modify: `phosp/pipeline.py` — atomic writes, orphan cleanup, dependency check, disk warning, `dry_run` support, ui hooks
- Modify: `tests/test_pipeline.py` — new tests + patch `shutil.which` in existing tests

**Interfaces:**
- Consumes: `PhospUI` from `phosp/ui.py` (Task 2)
- Produces:
  - `Stage.__init__(config, engine, forcefield, output_root, ui=None)`
  - `Pipeline.__init__(config, output_root, ui=None)`
  - `Pipeline.execute(start_from=None, only_stages=None, dry_run=False)`

---

- [ ] **Step 1: Write failing tests**

Replace `tests/test_pipeline.py` entirely:

```python
from __future__ import annotations
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from phosp.config import load_config
from phosp.exceptions import PhospError
from phosp.pipeline import Pipeline

FIXTURES = Path(__file__).parent / "fixtures"


def _make_pipeline(tmp_path: Path) -> Pipeline:
    cfg = load_config(FIXTURES / "valid_config.yaml")
    return Pipeline(cfg, output_root=tmp_path / "output")


def _patched_gmx():
    return patch("phosp.pipeline.shutil.which", return_value="/usr/bin/gmx")


def test_pipeline_creates_output_dir(tmp_path):
    p = _make_pipeline(tmp_path)
    assert (tmp_path / "output").exists()


def test_pipeline_skips_completed_stages(tmp_path):
    p = _make_pipeline(tmp_path)
    p.checkpoint.mark_complete("stage1", {"modified_pdb": "fake.pdb"})
    mock_stage = MagicMock()
    with _patched_gmx(), patch.object(p, "_build_stage", return_value=mock_stage):
        p.execute(only_stages="1")
    mock_stage.run.assert_not_called()


def test_start_from_skips_earlier_stages(tmp_path):
    p = _make_pipeline(tmp_path)
    called = []
    p._run_stage = lambda name: called.append(name)
    with _patched_gmx():
        p.execute(start_from="stage3", only_stages="1,2,3")
    assert "stage1" not in called
    assert "stage2" not in called


def test_dependency_check_raises_if_no_gmx(tmp_path):
    p = _make_pipeline(tmp_path)
    with patch("phosp.pipeline.shutil.which", return_value=None):
        with pytest.raises(PhospError, match="GROMACS"):
            p.execute(only_stages="1")


def test_dependency_check_passes_with_gmx(tmp_path):
    p = _make_pipeline(tmp_path)
    mock_stage = MagicMock()
    mock_stage.run.return_value = MagicMock(artifacts={})
    with _patched_gmx(), patch.object(p, "_build_stage", return_value=mock_stage):
        p.execute(only_stages="1")  # should not raise


def test_dry_run_does_not_execute_stages(tmp_path):
    p = _make_pipeline(tmp_path)
    with _patched_gmx():
        p.execute(only_stages="1", dry_run=True)
    assert not (tmp_path / "output" / "stage1").exists()
    assert not p.checkpoint.is_complete("stage1")


def test_atomic_success_renames_tmp_to_final(tmp_path):
    p = _make_pipeline(tmp_path)
    final_dir = tmp_path / "output" / "stage1"

    def fake_run():
        return MagicMock(artifacts={})

    mock_stage = MagicMock()
    mock_stage.run.side_effect = fake_run
    with _patched_gmx(), patch.object(p, "_build_stage", return_value=mock_stage):
        p.execute(only_stages="1")

    assert final_dir.exists()
    assert not (tmp_path / "output" / ".stage1_tmp").exists()


def test_atomic_failure_cleans_tmp(tmp_path):
    p = _make_pipeline(tmp_path)
    mock_stage = MagicMock()
    mock_stage.run.side_effect = RuntimeError("boom")

    with _patched_gmx(), patch.object(p, "_build_stage", return_value=mock_stage):
        with pytest.raises(RuntimeError, match="boom"):
            p.execute(only_stages="1")

    assert not (tmp_path / "output" / ".stage1_tmp").exists()
    assert not (tmp_path / "output" / "stage1").exists()


def test_orphan_tmp_dirs_removed_at_startup(tmp_path):
    p = _make_pipeline(tmp_path)
    orphan = tmp_path / "output" / ".stage1_tmp"
    orphan.mkdir(parents=True)

    mock_stage = MagicMock()
    mock_stage.run.return_value = MagicMock(artifacts={})
    with _patched_gmx(), patch.object(p, "_build_stage", return_value=mock_stage):
        p.execute(only_stages="1")

    assert not orphan.exists()


def test_disk_warning_logged_when_low(tmp_path, caplog):
    import logging
    cfg = load_config(FIXTURES / "valid_config.yaml")
    cfg.simulation.production_time_ns = 100.0
    p = Pipeline(cfg, output_root=tmp_path / "output")

    with _patched_gmx(), \
         patch("phosp.pipeline.shutil.disk_usage",
               return_value=MagicMock(free=1 * 1024 ** 3)), \
         caplog.at_level(logging.WARNING, logger="phosp"):
        p.execute(dry_run=True)

    assert any("disk space" in r.message.lower() for r in caplog.records)
```

- [ ] **Step 2: Run to confirm failures**

```bash
pytest tests/test_pipeline.py -v
```
Expected: new tests fail (`dry_run` not a param, no `shutil.which` call, etc.).

- [ ] **Step 3: Update `phosp/stages/base.py`**

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phosp.ui import PhospUI


@dataclass
class StageResult:
    stage: str
    output_dir: Path
    artifacts: dict[str, Path] = field(default_factory=dict)


class Stage(ABC):
    def __init__(self, config, engine, forcefield, output_root: Path, ui: PhospUI | None = None) -> None:
        self.config = config
        self.engine = engine
        self.forcefield = forcefield
        self.output_root = output_root
        self.ui = ui

    @abstractmethod
    def validate_inputs(self) -> None:
        """Raise StageInputError if preconditions are not met."""

    @abstractmethod
    def run(self) -> StageResult:
        """Execute the stage and return paths to produced artifacts."""
```

- [ ] **Step 4: Replace `phosp/pipeline.py`**

```python
from __future__ import annotations
import logging
import shutil
import time
from pathlib import Path

from phosp.config import PhospConfig
from phosp.engines.gromacs import GROMACSEngine
from phosp.exceptions import PhospError
from phosp.utils.checkpoint import Checkpoint

logger = logging.getLogger(__name__)

_ALL_STAGES = ["stage1", "stage2", "stage3", "stage4"]


class Pipeline:
    def __init__(self, config: PhospConfig, output_root: Path, ui=None) -> None:
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
        self._preflight_checks()
        if dry_run:
            return
        self._clean_orphan_tmpdirs()
        stages = self._resolve_stages(start_from, only_stages)
        for stage_name in stages:
            if self.checkpoint.is_complete(stage_name):
                logger.info("Skipping %s (already complete)", stage_name)
                continue
            self._run_stage(stage_name)

    def _preflight_checks(self) -> None:
        if shutil.which("gmx") is None:
            raise PhospError(
                "GROMACS (gmx) not found in PATH. "
                "Install GROMACS and ensure gmx is on your PATH."
            )
        self._warn_disk_space()

    def _warn_disk_space(self) -> None:
        estimated_gb = self.config.simulation.production_time_ns * 1.0 + 0.5
        try:
            available_gb = shutil.disk_usage(self.output_root).free / (1024 ** 3)
            if available_gb < estimated_gb:
                logger.warning(
                    "Low disk space: estimated %.1f GB needed, %.1f GB available at %s",
                    estimated_gb, available_gb, self.output_root,
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
        if start_from:
            try:
                stages = stages[stages.index(start_from):]
            except ValueError:
                raise PhospError(f"Unknown stage: {start_from}")
        return stages

    def _run_stage(self, stage_name: str) -> None:
        from phosp.forcefields.charmm36m import CHARMM36mFF
        from phosp.forcefields.amber_ff14sb import AMBERff14SBFF

        final_dir = self.output_root / stage_name
        tmp_dir = self.output_root / f".{stage_name}_tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        engine = GROMACSEngine()
        ff = CHARMM36mFF() if self.config.forcefield == "charmm36m" else AMBERff14SBFF()
        stage = self._build_stage(stage_name, engine, ff, tmp_dir)

        if self.ui:
            self.ui.stage_start(stage_name)

        start = time.monotonic()
        try:
            stage.validate_inputs()
            result = stage.run()
            tmp_dir.rename(final_dir)
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
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_pipeline.py tests/test_integration.py -v
```
Expected: all pass. The integration tests patch `GROMACSEngine` directly so they don't need `shutil.which` patched — but verify they still pass after this change. If they fail because `_preflight_checks` calls `shutil.which("gmx")`, add `patch("phosp.pipeline.shutil.which", return_value="/usr/bin/gmx")` to the integration test patches.

- [ ] **Step 6: Fix integration tests if needed**

If `tests/test_integration.py` fails with `PhospError: GROMACS (gmx) not found`, update each `with patch(...)` block to also include `patch("phosp.pipeline.shutil.which", return_value="/usr/bin/gmx")`.

- [ ] **Step 7: Run full suite**

```bash
pytest tests/ -q
```
Expected: 60+ passed.

- [ ] **Step 8: Commit**

```bash
git add phosp/stages/base.py phosp/pipeline.py tests/test_pipeline.py tests/test_integration.py
git commit -m "feat: atomic stage writes, dependency check, disk warning, dry-run support"
```

---

### Task 5: CLI Enhancements

**Files:**
- Modify: `phosp/cli.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: `configure_logging` (Task 1), `PhospUI` (Task 2), `Pipeline.execute(dry_run=)` (Task 4)
- Produces:
  - `phosp init [path]` — writes starter config YAML
  - `phosp status <output_dir>` — shows checkpoint table, exits 0/1
  - `phosp run` — gains `--dry-run`, `--log-level`, `--log-file`
  - `phosp validate`, `phosp predict-sites`, `phosp report` — call `configure_logging()` at startup

---

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cli.py
from __future__ import annotations
import json
from pathlib import Path
from typer.testing import CliRunner
from phosp.cli import app

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"


def test_init_creates_config_file(tmp_path: Path):
    out = tmp_path / "config.yaml"
    result = runner.invoke(app, ["init", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    content = out.read_text()
    assert "source:" in content
    assert "modification:" in content


def test_init_refuses_existing_file(tmp_path: Path):
    out = tmp_path / "config.yaml"
    out.write_text("existing: true\n")
    result = runner.invoke(app, ["init", str(out)])
    assert result.exit_code == 1
    assert out.read_text() == "existing: true\n"


def test_status_all_complete_exits_0(tmp_path: Path):
    out = tmp_path / "output"
    out.mkdir()
    checkpoint = {
        "completed_stages": ["stage1", "stage2", "stage3", "stage4"],
        "artifacts": {"stage1": {"modified_pdb": str(out / "stage1" / "modified.pdb")}},
        "stage1_completed_at": "2026-01-01T00:00:00",
        "stage2_completed_at": "2026-01-01T01:00:00",
        "stage3_completed_at": "2026-01-01T10:00:00",
        "stage4_completed_at": "2026-01-01T11:00:00",
    }
    (out / "checkpoint.json").write_text(json.dumps(checkpoint))
    result = runner.invoke(app, ["status", str(out)])
    assert result.exit_code == 0


def test_status_partial_exits_1(tmp_path: Path):
    out = tmp_path / "output"
    out.mkdir()
    (out / "checkpoint.json").write_text(json.dumps({
        "completed_stages": ["stage1"],
        "artifacts": {},
    }))
    result = runner.invoke(app, ["status", str(out)])
    assert result.exit_code == 1


def test_status_missing_checkpoint_exits_1(tmp_path: Path):
    out = tmp_path / "output"
    out.mkdir()
    result = runner.invoke(app, ["status", str(out)])
    assert result.exit_code == 1


def test_validate_accepts_valid_config():
    result = runner.invoke(app, ["validate", str(FIXTURES / "valid_config.yaml")])
    assert result.exit_code == 0
    assert "valid" in result.output.lower()
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_cli.py -v
```
Expected: `test_init_*` and `test_status_*` fail (`No such command 'init'`, `'status'`).

- [ ] **Step 3: Write the starter config constant and replace `phosp/cli.py`**

```python
# phosp/cli.py
from __future__ import annotations
from pathlib import Path
import typer

app = typer.Typer(help="Automated phosphorylation + MD simulation pipeline")

_STARTER_CONFIG = """\
# phosp configuration — run `phosp validate <this-file>` to check it.

input:
  source: pdb           # "pdb" or "uniprot"
  path: protein.pdb     # path to PDB file (required when source=pdb)
  # uniprot_id: P12345  # UniProt accession (required when source=uniprot)
  ph: 7.4               # pH for protonation state assignment

modification:
  sites:
    - chain: A          # PDB chain ID
      resid: 42         # residue number
      resname: SER      # SER, THR, or TYR
      phospho_type: pSer  # pSer, pThr, or pTyr (must match resname)

forcefield: charmm36m   # "charmm36m" or "amber_ff14sb"
protocol: globular_protein  # "globular_protein", "membrane_protein", or "phosphopeptide"

simulation:
  production_time_ns: 100.0   # production run length in ns
  output_freq_ps: 10.0        # trajectory output frequency in ps
  water_model: tip3p          # "tip3p" or "spce"
  box_type: dodecahedron      # "dodecahedron" or "cubic"
  salt_concentration_mM: 150.0

  hpc:
    enabled: false        # true to generate HPC job scripts
    scheduler: slurm      # "slurm" or "pbs"
    ntasks: 8
    gpus: 1
    walltime: "24:00:00"
    partition: gpu
    auto_submit: false

analysis:
  plugins:
    - rmsd
    - rmsf
    - radius_of_gyration
    - secondary_structure
    - hbond
    - contacts
    - sasa
  rmsd:
    selection: backbone
  rmsf:
    selection: name CA
  sasa:
    residues: []

# Next: phosp predict-sites <this-file>
"""


@app.command()
def run(
    config_path: Path = typer.Argument(..., help="Path to config YAML"),
    start_from: str = typer.Option(None, "--start-from", help="stage1|stage2|stage3|stage4"),
    stages: str = typer.Option(None, "--stages", help="e.g. '1,2'"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config and environment without running"),
    log_level: str = typer.Option("INFO", "--log-level", help="DEBUG|INFO|WARNING|ERROR"),
    log_file: Path = typer.Option(None, "--log-file", help="Write logs to this file"),
) -> None:
    from phosp.config import load_config
    from phosp.logging import configure_logging
    from phosp.pipeline import Pipeline
    from phosp.ui import PhospUI

    configure_logging(level=log_level, log_file=log_file)
    cfg = load_config(config_path)

    if dry_run:
        if cfg.input.source == "pdb" and cfg.input.path and not cfg.input.path.exists():
            typer.echo(f"Error: input PDB not found: {cfg.input.path}", err=True)
            raise typer.Exit(code=1)
        Pipeline(cfg, output_root=config_path.parent / "output").execute(dry_run=True)
        estimated_gb = cfg.simulation.production_time_ns * 1.0 + 0.5
        typer.echo(f"Estimated disk space needed: {estimated_gb:.1f} GB")
        typer.echo("Dry run complete — no stages executed")
        return

    ui = PhospUI()
    Pipeline(cfg, output_root=config_path.parent / "output", ui=ui).execute(
        start_from=start_from, only_stages=stages
    )


@app.command()
def validate(config_path: Path = typer.Argument(...)) -> None:
    from phosp.config import load_config
    from phosp.logging import configure_logging
    configure_logging()
    load_config(config_path)
    typer.echo("Config valid.")


@app.command(name="predict-sites")
def predict_sites(
    config_path: Path = typer.Argument(...),
    threshold: float = typer.Option(0.5, "--threshold"),
) -> None:
    from phosp.config import load_config
    from phosp.logging import configure_logging
    from phosp.prediction.netphos import NetPhos
    configure_logging()
    cfg = load_config(config_path)
    results = NetPhos().predict(cfg.input.path, threshold=threshold)
    for r in results:
        typer.echo(f"  chain={r['chain']} resid={r['resid']} resname={r['resname']} "
                   f"type={r['phospho_type']} score={r['score']:.3f}")
    typer.echo(f"\nAdd selected entries to modification.sites in {config_path}")


@app.command()
def report(output_dir: Path = typer.Argument(...)) -> None:
    from phosp.logging import configure_logging
    from phosp.stages.stage4_analyze import Stage4Analyze
    configure_logging()
    Stage4Analyze.regenerate_report(output_dir)
    typer.echo(f"Report written to {output_dir}/report.html")


@app.command()
def init(
    path: Path = typer.Argument(Path("phosp_config.yaml"), help="Output path for the config file"),
) -> None:
    from phosp.logging import configure_logging
    configure_logging()
    if path.exists():
        typer.echo(f"Error: {path} already exists. Use a different path or delete it first.", err=True)
        raise typer.Exit(code=1)
    path.write_text(_STARTER_CONFIG)
    typer.echo(f"Config written to {path}")
    typer.echo(f"Next: phosp predict-sites {path}")


@app.command()
def status(output_dir: Path = typer.Argument(..., help="Pipeline output directory")) -> None:
    import json
    from rich.console import Console
    from rich.table import Table
    from phosp.logging import configure_logging
    configure_logging()

    checkpoint_path = output_dir / "checkpoint.json"
    if not checkpoint_path.exists():
        typer.echo(f"Error: no checkpoint found at {checkpoint_path}", err=True)
        raise typer.Exit(code=1)

    data = json.loads(checkpoint_path.read_text())
    completed = set(data.get("completed_stages", []))

    _labels = {
        "stage1": "Stage 1 — Chemical Modification",
        "stage2": "Stage 2 — MD Preparation",
        "stage3": "Stage 3 — MD Simulation",
        "stage4": "Stage 4 — Analysis",
    }

    console = Console()
    table = Table(title="phosp Pipeline Status")
    table.add_column("Stage", style="cyan", no_wrap=True)
    table.add_column("Status")
    table.add_column("Completed At")
    table.add_column("Key Artifacts")

    for s in ["stage1", "stage2", "stage3", "stage4"]:
        if s in completed:
            status_str = "[green]✓ complete[/]"
            completed_at = data.get(f"{s}_completed_at", "")
            artifacts = data.get("artifacts", {}).get(s, {})
            artifact_str = ", ".join(Path(v).name for v in list(artifacts.values())[:3])
        else:
            status_str = "[dim]pending[/]"
            completed_at = ""
            artifact_str = ""
        table.add_row(_labels[s], status_str, completed_at, artifact_str)

    console.print(table)

    if len(completed) < 4:
        raise typer.Exit(code=1)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_cli.py -v
```
Expected: `6 passed`

- [ ] **Step 5: Run full suite**

```bash
pytest tests/ -q
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add phosp/cli.py tests/test_cli.py
git commit -m "feat: phosp init, status commands; --dry-run, --log-level, --log-file flags"
```

---

### Task 6: Plugin Fail-Soft + Report Warning Banner

**Files:**
- Modify: `phosp/stages/stage4_analyze.py`
- Modify: `phosp/templates/report.html.j2`
- Modify: `tests/test_stage4.py`

**Interfaces:**
- Consumes: `self.ui` (from `Stage` base, Task 4) — calls `self.ui.plugin_start(name)` when set
- Produces:
  - `Stage4Analyze.run()` — collects failures per plugin; raises `AnalysisError` only if all plugins fail; generates HTML report with warning banner for failed plugins
  - `_render_report(output_dir, failed_plugins=[])` — gains `failed_plugins: list[tuple[str, str]]` param
  - `Stage4Analyze.regenerate_report(output_dir)` — passes `failed_plugins=[]`

---

- [ ] **Step 1: Write failing tests**

Add to `tests/test_stage4.py`:

```python
def test_plugin_partial_failure_continues(tmp_path):
    """One plugin fails, others succeed — no exception, partial results saved."""
    from phosp.exceptions import AnalysisError

    cfg = load_config(FIXTURES / "valid_config.yaml")
    cfg.analysis.plugins = ["bad", "fake"]

    stage3_dir = tmp_path / "output" / "stage3" / "production"
    stage3_dir.mkdir(parents=True)
    (stage3_dir / "production.xtc").write_bytes(b"")
    (stage3_dir / "production.tpr").write_bytes(b"")

    class _BadPlugin(AnalysisPlugin):
        name = "bad"
        def run(self, universe, config):
            raise RuntimeError("intentional failure")
        def plot(self, result):
            import matplotlib.pyplot as plt
            return plt.figure()

    stage = Stage4Analyze(cfg, MagicMock(), MagicMock(), tmp_path / "output" / "stage4")
    fake_universe = MagicMock()
    with patch("phosp.stages.stage4_analyze.mda.Universe", return_value=fake_universe), \
         patch("phosp.stages.stage4_analyze._discover_plugins",
               return_value={"bad": _BadPlugin, "fake": _FakePlugin}):
        result = stage.run()

    assert (tmp_path / "output" / "stage4" / "fake.csv").exists()
    assert not (tmp_path / "output" / "stage4" / "bad.csv").exists()


def test_all_plugins_fail_raises_analysis_error(tmp_path):
    """All plugins fail → AnalysisError raised."""
    from phosp.exceptions import AnalysisError

    cfg = load_config(FIXTURES / "valid_config.yaml")
    cfg.analysis.plugins = ["bad"]

    stage3_dir = tmp_path / "output" / "stage3" / "production"
    stage3_dir.mkdir(parents=True)
    (stage3_dir / "production.xtc").write_bytes(b"")
    (stage3_dir / "production.tpr").write_bytes(b"")

    class _BadPlugin(AnalysisPlugin):
        name = "bad"
        def run(self, universe, config):
            raise RuntimeError("all dead")
        def plot(self, result):
            import matplotlib.pyplot as plt
            return plt.figure()

    stage = Stage4Analyze(cfg, MagicMock(), MagicMock(), tmp_path / "output" / "stage4")
    fake_universe = MagicMock()
    with patch("phosp.stages.stage4_analyze.mda.Universe", return_value=fake_universe), \
         patch("phosp.stages.stage4_analyze._discover_plugins",
               return_value={"bad": _BadPlugin}):
        with pytest.raises(AnalysisError, match="All analysis plugins failed"):
            stage.run()


def test_report_generated_after_run(tmp_path):
    """run() generates report.html."""
    cfg = load_config(FIXTURES / "valid_config.yaml")
    cfg.analysis.plugins = ["fake"]

    stage3_dir = tmp_path / "output" / "stage3" / "production"
    stage3_dir.mkdir(parents=True)
    (stage3_dir / "production.xtc").write_bytes(b"")
    (stage3_dir / "production.tpr").write_bytes(b"")

    stage = Stage4Analyze(cfg, MagicMock(), MagicMock(), tmp_path / "output" / "stage4")
    fake_universe = MagicMock()
    with patch("phosp.stages.stage4_analyze.mda.Universe", return_value=fake_universe), \
         patch("phosp.stages.stage4_analyze._discover_plugins",
               return_value={"fake": _FakePlugin}):
        stage.run()

    assert (tmp_path / "output" / "stage4" / "report.html").exists()
```

- [ ] **Step 2: Run to confirm failures**

```bash
pytest tests/test_stage4.py -v
```
Expected: the three new tests fail.

- [ ] **Step 3: Update `phosp/stages/stage4_analyze.py`**

Replace the file:

```python
from __future__ import annotations
import importlib
import logging
import pkgutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda

import phosp.plugins.analysis as _analysis_pkg
from phosp.exceptions import AnalysisError, StageInputError
from phosp.plugins.analysis.base import AnalysisPlugin
from phosp.stages.base import Stage, StageResult

logger = logging.getLogger(__name__)


def _discover_plugins() -> dict[str, type[AnalysisPlugin]]:
    registry: dict[str, type[AnalysisPlugin]] = {}
    pkg_path = Path(_analysis_pkg.__file__).parent
    for _, mod_name, _ in pkgutil.iter_modules([str(pkg_path)]):
        if mod_name == "base":
            continue
        importlib.import_module(f"phosp.plugins.analysis.{mod_name}")
    for cls in AnalysisPlugin.__subclasses__():
        registry[cls.name] = cls
    return registry


class Stage4Analyze(Stage):
    def validate_inputs(self) -> None:
        prod_dir = self.output_root.parent / "stage3" / "production"
        for f in ["production.xtc", "production.tpr"]:
            if not (prod_dir / f).exists():
                raise StageInputError(
                    f"{f} not found in {prod_dir}. Run stage3 first."
                )

    def run(self) -> StageResult:
        out = self.output_root
        out.mkdir(parents=True, exist_ok=True)
        cfg = self.config
        prod_dir = out.parent / "stage3" / "production"

        universe = mda.Universe(
            str(prod_dir / "production.tpr"),
            str(prod_dir / "production.xtc"),
        )

        registry = _discover_plugins()
        requested = cfg.analysis.plugins
        artifacts: dict[str, Path] = {}
        failures: list[tuple[str, str]] = []

        for plugin_name in requested:
            if plugin_name not in registry:
                logger.warning("Plugin '%s' not found — skipping", plugin_name)
                continue
            plugin_config = getattr(cfg.analysis, plugin_name, {})
            if not isinstance(plugin_config, dict):
                plugin_config = plugin_config.model_dump() if hasattr(plugin_config, "model_dump") else {}

            if self.ui:
                self.ui.plugin_start(plugin_name)

            try:
                plugin = registry[plugin_name]()
                result_df = plugin.run(universe, plugin_config)
                csv_path = out / f"{plugin_name}.csv"
                result_df.to_csv(csv_path, index=False)

                fig = plugin.plot(result_df)
                png_path = out / f"{plugin_name}.png"
                fig.savefig(png_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

                artifacts[plugin_name] = csv_path
                logger.info("Plugin '%s' complete", plugin_name)
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                logger.warning("Plugin '%s' failed: %s", plugin_name, error_msg)
                failures.append((plugin_name, error_msg))

        if failures and not artifacts:
            raise AnalysisError(
                "All analysis plugins failed:\n"
                + "\n".join(f"  {n}: {e}" for n, e in failures)
            )

        _render_report(out, failed_plugins=failures)
        return StageResult(stage="stage4", output_dir=out, artifacts=artifacts)

    @staticmethod
    def regenerate_report(output_dir: Path) -> None:
        _render_report(output_dir, failed_plugins=[])


def _render_report(
    output_dir: Path,
    failed_plugins: list[tuple[str, str]] | None = None,
) -> Path:
    from jinja2 import Environment, FileSystemLoader
    import base64
    templates_dir = Path(__file__).parent.parent / "templates"
    env = Environment(loader=FileSystemLoader(str(templates_dir)))
    template = env.get_template("report.html.j2")

    png_files = sorted(output_dir.glob("*.png"))
    figures = []
    for png in png_files:
        b64 = base64.b64encode(png.read_bytes()).decode()
        figures.append({"name": png.stem, "data": b64})

    html = template.render(
        figures=figures,
        output_dir=str(output_dir),
        failed_plugins=failed_plugins or [],
    )
    report_path = output_dir / "report.html"
    report_path.write_text(html)
    return report_path
```

- [ ] **Step 4: Update `phosp/templates/report.html.j2`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Phosp Analysis Report</title>
  <style>
    body { font-family: sans-serif; max-width: 1100px; margin: auto; padding: 2rem; }
    h1 { color: #2c3e50; }
    .figure-section { margin-bottom: 3rem; }
    .figure-section h2 { border-bottom: 1px solid #ccc; padding-bottom: 0.3rem; }
    img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
    .failed-plugins { background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; padding: 1rem 1.5rem; margin-bottom: 2rem; }
    .failed-plugins h2 { color: #856404; margin-top: 0; }
    .plugin-error { margin: 0.4rem 0; font-family: monospace; font-size: 0.9em; }
  </style>
</head>
<body>
  <h1>Phosp MD Analysis Report</h1>
  <p>Output directory: <code>{{ output_dir }}</code></p>

  {% if failed_plugins %}
  <div class="failed-plugins">
    <h2>⚠ Failed Plugins</h2>
    {% for name, error in failed_plugins %}
    <div class="plugin-error"><strong>{{ name }}</strong>: {{ error }}</div>
    {% endfor %}
  </div>
  {% endif %}

  {% for fig in figures %}
  <div class="figure-section">
    <h2>{{ fig.name | replace("_", " ") | title }}</h2>
    <img src="data:image/png;base64,{{ fig.data }}" alt="{{ fig.name }}">
  </div>
  {% endfor %}
</body>
</html>
```

- [ ] **Step 5: Run stage4 tests**

```bash
pytest tests/test_stage4.py -v
```
Expected: all pass (existing 3 + new 3 = 6 total).

- [ ] **Step 6: Run full suite**

```bash
pytest tests/ -q
```
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add phosp/stages/stage4_analyze.py phosp/templates/report.html.j2 tests/test_stage4.py
git commit -m "feat: plugin fail-soft in stage4, warning banner in HTML report"
```

---

## Self-Review Checklist

**Spec coverage:**
- §1 Atomic Stage Writes → Task 4 (`_run_stage` tmp dir, rename, cleanup)
- §1 Orphan cleanup → Task 4 (`_clean_orphan_tmpdirs`)
- §2 PhospUI class → Task 2
- §2 `phosp status` command → Task 5
- §2 Stage 4 plugin progress → Task 6 (`self.ui.plugin_start`)
- §3a Dependency check → Task 4 (`_preflight_checks`)
- §3b Config validators → Task 3
- §3c Disk space warning → Task 4 (`_warn_disk_space`)
- §3d `phosp init` → Task 5
- §3e `--dry-run` → Tasks 4 + 5
- §4a Plugin fail-soft → Task 6
- §4b Logging configuration → Task 1
- §4b `--log-level` / `--log-file` → Task 5

**Files added/modified summary:**

| File | Task |
|------|------|
| `phosp/logging.py` (new) | 1 |
| `phosp/ui.py` (new) | 2 |
| `phosp/config.py` | 3 |
| `phosp/stages/base.py` | 4 |
| `phosp/pipeline.py` | 4 |
| `phosp/cli.py` | 5 |
| `phosp/stages/stage4_analyze.py` | 6 |
| `phosp/templates/report.html.j2` | 6 |
