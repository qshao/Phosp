# Robustness + DX Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 7 independent robustness and developer-experience improvements: upfront stage/plugin validation, subprocess timeout, configurable pdb2pqr, config-hash guard, elapsed-time display, and GitHub Actions CI.

**Architecture:** Each item is a self-contained change. Tasks 1–2 are pure validation additions (no new config). Tasks 3–4 add config fields to `GROMACSConfig`. Task 5 adds a hash field to `Checkpoint` and a `config_path` parameter to `Pipeline`. Task 6 adds start/duration tracking to `Checkpoint` and a new column to the CLI status table. Task 7 is a new YAML file with no code deps.

**Tech Stack:** Python 3.10+, Pydantic v2, Typer + Rich, pytest, subprocess, hashlib, GitHub Actions

## Global Constraints

- Python ≥ 3.10 (match scripts use); Pydantic v2 throughout
- All new config fields optional with sane defaults — no breaking changes to existing YAML
- All new code covered by tests; tests use existing `pytest` patterns in `tests/`; fixture config at `tests/fixtures/valid_config.yaml`
- No new dependencies
- Commit after each task; message format: `feat: <short description>`

---

### Task 1: Upfront validation of `--stages` and `--start-from`

**Files:**
- Modify: `phosp/pipeline.py` — `_resolve_stages` method (lines 125–135)
- Modify: `tests/test_pipeline.py` — add two new tests

**Interfaces:**
- No interface changes; `_resolve_stages` signature unchanged
- Produces: `PhospError` raised before any stage executes when stage names are invalid

- [ ] **Step 1: Write the failing tests**

```python
# In tests/test_pipeline.py, add after existing tests:

def test_resolve_stages_raises_on_unknown_only_stages(tmp_path):
    """--stages with a bad stage name raises PhospError before any stage runs."""
    p = _make_pipeline(tmp_path)
    with pytest.raises(PhospError, match="Unknown stage"):
        p._resolve_stages(start_from=None, only_stages="1,foo")


def test_resolve_stages_raises_on_unknown_start_from(tmp_path):
    """--start-from with a bad name raises PhospError before any stage runs."""
    p = _make_pipeline(tmp_path)
    with pytest.raises(PhospError, match="Unknown stage"):
        p._resolve_stages(start_from="stageX", only_stages=None)
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_pipeline.py::test_resolve_stages_raises_on_unknown_only_stages \
       tests/test_pipeline.py::test_resolve_stages_raises_on_unknown_start_from -v
```
Expected: FAIL — no `PhospError` is raised for `"foo"` in `only_stages` today.

- [ ] **Step 3: Replace `_resolve_stages` in `phosp/pipeline.py`**

Replace the current method body (lines 125–135) with:

```python
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
```

- [ ] **Step 4: Run all tests**

```
pytest tests/test_pipeline.py -v
```
Expected: all pass including the two new tests.

- [ ] **Step 5: Commit**

```bash
git add phosp/pipeline.py tests/test_pipeline.py
git commit -m "feat: upfront validation of --stages and --start-from"
```

---

### Task 2: Unknown analysis plugin names → immediate error

**Files:**
- Modify: `phosp/stages/stage4_analyze.py` — `Stage4Analyze.run` (around line 66)
- Modify: `tests/test_stage4.py` — add one new test

**Interfaces:**
- Raises `AnalysisError` before MDAnalysis Universe is loaded when any plugin name is unknown
- Existing tests that use `_discover_plugins` mocked with all requested names present are unaffected

- [ ] **Step 1: Write the failing test**

```python
# In tests/test_stage4.py, add after existing tests:

def test_unknown_plugin_name_raises_analysis_error(tmp_path):
    """A typo in analysis.plugins raises AnalysisError immediately with valid names listed."""
    from phosp.exceptions import AnalysisError
    cfg = load_config(FIXTURES / "valid_config.yaml")
    cfg.analysis.plugins = ["rmsdf"]   # typo: should be "rmsf"

    stage3_dir = tmp_path / "output" / "stage3" / "production"
    stage3_dir.mkdir(parents=True)
    (stage3_dir / "production.xtc").write_bytes(b"")
    (stage3_dir / "production.gro").write_bytes(b"")

    stage = Stage4Analyze(cfg, MagicMock(), MagicMock(), tmp_path / "output" / "stage4")
    with patch("phosp.stages.stage4_analyze._discover_plugins",
               return_value={"rmsd": MagicMock, "rmsf": MagicMock}), \
         patch("phosp.stages.stage4_analyze.mda.Universe", return_value=MagicMock()):
        with pytest.raises(AnalysisError, match="Unknown analysis plugins.*rmsdf"):
            stage.run()
```

- [ ] **Step 2: Run test to confirm it fails**

```
pytest tests/test_stage4.py::test_unknown_plugin_name_raises_analysis_error -v
```
Expected: FAIL — the typo is currently silently skipped.

- [ ] **Step 3: Add the unknown-plugin check in `Stage4Analyze.run`**

In `phosp/stages/stage4_analyze.py`, in the `run` method, immediately after `registry = _discover_plugins()` and before the `requested = cfg.analysis.plugins` line:

```python
registry = _discover_plugins()
requested = cfg.analysis.plugins
unknown = [p for p in requested if p not in registry]
if unknown:
    raise AnalysisError(
        f"Unknown analysis plugins: {unknown}. "
        f"Valid plugins: {sorted(registry)}"
    )
```

The file currently has `registry = _discover_plugins()` on line 66 and `requested = cfg.analysis.plugins` on line 67. Replace those two lines with the four lines above.

- [ ] **Step 4: Run all stage4 tests**

```
pytest tests/test_stage4.py -v
```
Expected: all pass including the new test. Confirm `test_plugin_partial_failure_continues` still passes — that test patches `_discover_plugins` to include both "bad" and "fake" so the early check passes.

- [ ] **Step 5: Commit**

```bash
git add phosp/stages/stage4_analyze.py tests/test_stage4.py
git commit -m "feat: unknown analysis plugin names raise AnalysisError immediately"
```

---

### Task 3: Subprocess timeout for GROMACS commands

**Files:**
- Modify: `phosp/config.py` — add `timeout_minutes` field to `GROMACSConfig`
- Modify: `phosp/engines/gromacs.py` — `_run_gmx` and `GROMACSEngine`
- Modify: `phosp/pipeline.py` — pass `timeout_minutes` to `GROMACSEngine`
- Modify: `phosp/cli.py` — add `timeout_minutes` to starter config comment
- Modify: `tests/test_gromacs_engine.py` — add timeout tests

**Interfaces:**
- `_run_gmx(cmd, cwd, input_text="", timeout=None)` — `timeout` in seconds (int | None)
- `GROMACSEngine(binary="gmx", timeout_minutes=None)` — new kwarg
- `GROMACSConfig.timeout_minutes: int | None = None`

- [ ] **Step 1: Write the failing tests**

```python
# In tests/test_gromacs_engine.py, add:

def test_run_gmx_raises_simulation_error_on_timeout(tmp_path):
    """TimeoutExpired is caught and re-raised as SimulationError with 'timed out'."""
    import subprocess
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["gmx"], 60)):
        with pytest.raises(SimulationError, match="timed out after 1 minutes"):
            _run_gmx(["gmx", "mdrun"], cwd=tmp_path, timeout=60)


def test_engine_passes_timeout_to_run_gmx(tmp_path):
    """GROMACSEngine with timeout_minutes=2 passes 120s to _run_gmx."""
    engine = GROMACSEngine(binary="gmx", timeout_minutes=2)
    phase_dir = tmp_path / "min"
    phase_dir.mkdir()
    (phase_dir / "min.log").write_text("done")

    with patch("phosp.engines.gromacs._run_gmx") as mock_gmx:
        mock_gmx.return_value = MagicMock(returncode=0)
        engine.run_phase(
            phase="min",
            mdp=tmp_path / "min.mdp",
            topology=tmp_path / "topol.top",
            structure=tmp_path / "ions.gro",
            output_dir=phase_dir,
        )
    for call in mock_gmx.call_args_list:
        assert call.kwargs.get("timeout") == 120


def test_engine_no_timeout_by_default(tmp_path):
    """GROMACSEngine with no timeout_minutes passes timeout=None to _run_gmx."""
    engine = GROMACSEngine()
    phase_dir = tmp_path / "min"
    phase_dir.mkdir()
    (phase_dir / "min.log").write_text("done")

    with patch("phosp.engines.gromacs._run_gmx") as mock_gmx:
        mock_gmx.return_value = MagicMock(returncode=0)
        engine.run_phase(
            phase="min",
            mdp=tmp_path / "min.mdp",
            topology=tmp_path / "topol.top",
            structure=tmp_path / "ions.gro",
            output_dir=phase_dir,
        )
    for call in mock_gmx.call_args_list:
        assert call.kwargs.get("timeout") is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_gromacs_engine.py::test_run_gmx_raises_simulation_error_on_timeout \
       tests/test_gromacs_engine.py::test_engine_passes_timeout_to_run_gmx \
       tests/test_gromacs_engine.py::test_engine_no_timeout_by_default -v
```
Expected: FAIL — `_run_gmx` doesn't accept `timeout`, `GROMACSEngine` doesn't accept `timeout_minutes`.

- [ ] **Step 3: Update `GROMACSConfig` in `phosp/config.py`**

In the `GROMACSConfig` class (currently line 122–123), replace with:

```python
class GROMACSConfig(BaseModel):
    binary: str = "gmx"
    timeout_minutes: int | None = None  # None = no limit; e.g. 120 for 2-hour hard cap
```

- [ ] **Step 4: Update `_run_gmx` and `GROMACSEngine` in `phosp/engines/gromacs.py`**

Replace the current `_run_gmx` function:

```python
def _run_gmx(
    cmd: list[str], cwd: Path, input_text: str = "", timeout: int | None = None
) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True,
            input=input_text, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        minutes = timeout // 60 if timeout else "?"
        raise SimulationError(
            f"GROMACS timed out after {minutes} minutes: {cmd[0]} {cmd[1]}"
        )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout)[-2000:]
        raise SimulationError(f"GROMACS command failed: {' '.join(cmd)}\n{tail}")
    return result
```

Replace the `GROMACSEngine.__init__`:

```python
def __init__(self, binary: str = "gmx", timeout_minutes: int | None = None) -> None:
    self._binary = binary
    self._timeout = timeout_minutes * 60 if timeout_minutes is not None else None
```

In every `_run_gmx(...)` call inside `GROMACSEngine`, add `timeout=self._timeout` as a keyword argument. There are 6 calls total:
- `prepare_topology`: `_run_gmx(..., cwd=output_dir)` → `_run_gmx(..., cwd=output_dir, timeout=self._timeout)`
- `solvate` (2 calls): each gets `timeout=self._timeout`
- `add_ions` (2 calls): each gets `timeout=self._timeout`
- `run_phase` (2 calls — grompp and mdrun): each gets `timeout=self._timeout`

- [ ] **Step 5: Update `Pipeline._run_stage` in `phosp/pipeline.py`**

Replace line 147:
```python
engine = GROMACSEngine(binary=self.config.gromacs.binary)
```
with:
```python
engine = GROMACSEngine(
    binary=self.config.gromacs.binary,
    timeout_minutes=self.config.gromacs.timeout_minutes,
)
```

- [ ] **Step 6: Run all tests**

```
pytest tests/ -v --tb=short
```
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add phosp/config.py phosp/engines/gromacs.py phosp/pipeline.py tests/test_gromacs_engine.py
git commit -m "feat: configurable subprocess timeout for GROMACS commands"
```

---

### Task 4: Configurable pdb2pqr binary path

**Files:**
- Modify: `phosp/config.py` — add `pdb2pqr` field to `GROMACSConfig`
- Modify: `phosp/utils/structure.py` — add `pdb2pqr_binary` kwarg to `protonate_structure`
- Modify: `phosp/stages/stage1_modify.py` — pass `cfg.gromacs.pdb2pqr`
- Modify: `phosp/pipeline.py` — check `cfg.gromacs.pdb2pqr` in `_preflight_checks`
- Modify: `phosp/cli.py` — update `validate` command and starter config
- Add tests to `tests/test_cli.py` and a new `tests/test_structure.py` (or existing `test_config.py`)

**Interfaces:**
- `protonate_structure(pdb, output, ph=7.4, pdb2pqr_binary="pdb2pqr")` — new kwarg
- `GROMACSConfig.pdb2pqr: str = "pdb2pqr"` — name or full path

- [ ] **Step 1: Write the failing tests**

```python
# In tests/test_pipeline.py, add:

def test_preflight_checks_pdb2pqr_binary(tmp_path):
    """_preflight_checks uses cfg.gromacs.pdb2pqr, not a hardcoded string."""
    cfg = load_config(FIXTURES / "valid_config.yaml")
    cfg.gromacs.pdb2pqr = "/custom/pdb2pqr"
    p = Pipeline(cfg, output_root=tmp_path / "output")
    # The configured binary is missing, so preflight should raise
    with _patched_gmx(), \
         patch("phosp.pipeline.shutil.which", side_effect=lambda b: "/usr/bin/gmx" if b == "gmx" else None), \
         patch("phosp.pipeline.Path.is_file", return_value=False):
        with pytest.raises(PhospError, match="pdb2pqr"):
            p._preflight_checks()
```

```python
# New file: tests/test_structure.py

from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from phosp.utils.structure import protonate_structure


def test_protonate_structure_uses_configured_binary(tmp_path):
    """protonate_structure uses the pdb2pqr_binary kwarg, not hardcoded 'pdb2pqr'."""
    pdb = tmp_path / "in.pdb"
    pdb.write_text("ATOM ...")
    out = tmp_path / "out.pdb"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        protonate_structure(pdb, out, ph=7.4, pdb2pqr_binary="/opt/pdb2pqr/bin/pdb2pqr")
    
    called_cmd = mock_run.call_args.args[0]
    assert called_cmd[0] == "/opt/pdb2pqr/bin/pdb2pqr"


def test_protonate_structure_defaults_to_pdb2pqr(tmp_path):
    """protonate_structure defaults to 'pdb2pqr' when no binary specified."""
    pdb = tmp_path / "in.pdb"
    pdb.write_text("ATOM ...")
    out = tmp_path / "out.pdb"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        protonate_structure(pdb, out, ph=7.4)
    
    called_cmd = mock_run.call_args.args[0]
    assert called_cmd[0] == "pdb2pqr"
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_pipeline.py::test_preflight_checks_pdb2pqr_binary \
       tests/test_structure.py::test_protonate_structure_uses_configured_binary -v
```
Expected: FAIL — `GROMACSConfig` has no `pdb2pqr` attribute.

- [ ] **Step 3: Add `pdb2pqr` to `GROMACSConfig` in `phosp/config.py`**

```python
class GROMACSConfig(BaseModel):
    binary: str = "gmx"
    timeout_minutes: int | None = None
    pdb2pqr: str = "pdb2pqr"  # name or full path, e.g. "/opt/pdb2pqr/bin/pdb2pqr"
```

- [ ] **Step 4: Update `protonate_structure` in `phosp/utils/structure.py`**

Replace the function signature and first line of the command list:

```python
def protonate_structure(
    pdb: Path, output: Path, ph: float = 7.4, pdb2pqr_binary: str = "pdb2pqr"
) -> Path:
    pqr_output = output.with_suffix(".pqr")
    cmd = [
        pdb2pqr_binary,
        "--ff=CHARMM",
        "--titration-state-method=propka",
        f"--with-ph={ph}",
        "--pdb-output", str(output),
        str(pdb),
        str(pqr_output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"PDB2PQR failed:\n{result.stderr[-2000:]}")
    return output
```

- [ ] **Step 5: Pass configured binary from `Stage1Modify.run`**

In `phosp/stages/stage1_modify.py`, replace line 40:
```python
        protonated = protonate_structure(cleaned, out / "protonated.pdb", ph=cfg.input.ph)
```
with:
```python
        protonated = protonate_structure(
            cleaned, out / "protonated.pdb",
            ph=cfg.input.ph,
            pdb2pqr_binary=cfg.gromacs.pdb2pqr,
        )
```

- [ ] **Step 6: Update `_preflight_checks` in `phosp/pipeline.py`**

In `_preflight_checks` (currently lines 49–57), add a check for pdb2pqr after the GROMACS binary check:

```python
def _preflight_checks(self) -> None:
    binary = self.config.gromacs.binary
    if shutil.which(binary) is None and not Path(binary).is_file():
        raise PhospError(
            f"GROMACS binary '{binary}' not found. "
            "Set gromacs.binary in your config to the correct path or binary name."
        )
    pdb2pqr = self.config.gromacs.pdb2pqr
    if shutil.which(pdb2pqr) is None and not Path(pdb2pqr).is_file():
        raise PhospError(
            f"pdb2pqr binary '{pdb2pqr}' not found. "
            "Install it with 'pip install pdb2pqr' or set gromacs.pdb2pqr in your config."
        )
    self._check_forcefield()
    self._warn_disk_space()
```

- [ ] **Step 7: Update `validate` command in `phosp/cli.py`**

In the `validate` function, replace the hardcoded `shutil.which("pdb2pqr")` check (lines 157–159) with:

```python
    pdb2pqr = cfg.gromacs.pdb2pqr
    if shutil.which(pdb2pqr) is None and not Path(pdb2pqr).is_file():
        errors.append(
            f"pdb2pqr binary '{pdb2pqr}' not found — "
            "run: pip install pdb2pqr  (or set gromacs.pdb2pqr in config)"
        )
```

Also update the success echo from `typer.echo("  ✓ pdb2pqr found")` to `typer.echo(f"  ✓ {pdb2pqr} found")`.

Also update the starter config `_STARTER_CONFIG` to document the new field. In the `gromacs:` section, add a comment line:

```yaml
gromacs:
  binary: gmx           # gmx binary name or full path
  # pdb2pqr: pdb2pqr   # pdb2pqr binary name or full path; default is "pdb2pqr"
```

- [ ] **Step 8: Run all tests**

```
pytest tests/ -v --tb=short
```
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add phosp/config.py phosp/utils/structure.py phosp/stages/stage1_modify.py \
        phosp/pipeline.py phosp/cli.py tests/test_pipeline.py tests/test_structure.py
git commit -m "feat: configurable pdb2pqr binary path via gromacs.pdb2pqr"
```

---

### Task 5: Config hash guard in checkpoint

**Files:**
- Modify: `phosp/utils/checkpoint.py` — add `store_config_hash`, `get_config_hash`, update `mark_complete`
- Modify: `phosp/pipeline.py` — `Pipeline.__init__` gains `config_path`, `execute` checks hash
- Modify: `phosp/cli.py` — `run` command passes `config_path` to `Pipeline`
- Modify: `tests/test_pipeline.py` — add config hash test

**Interfaces:**
- `Checkpoint.mark_complete(stage, artifacts, config_hash=None)` — new optional kwarg, stored once
- `Checkpoint.get_config_hash() -> str | None`
- `Pipeline(config, output_root, ui=None, config_path=None)` — new optional kwarg

- [ ] **Step 1: Write the failing test**

```python
# In tests/test_pipeline.py, add at top: import shutil (already present), import logging
# Then add:

def test_config_hash_guard_warns_on_mismatch(tmp_path, caplog):
    """When config has changed since last run, a warning is logged."""
    import logging
    p_old = _make_pipeline(tmp_path)
    # Simulate prior run completing stage1 with a different hash
    p_old.checkpoint.mark_complete("stage1", {}, config_hash="oldhash12345678")

    # New pipeline points to config_path (real file) — its hash won't match "oldhash12345678"
    yaml_file = FIXTURES / "valid_config.yaml"
    cfg = load_config(yaml_file)
    p_new = Pipeline(cfg, output_root=tmp_path / "output", config_path=yaml_file)

    mock_stage = MagicMock()
    mock_stage.run.return_value = MagicMock(artifacts={})

    with caplog.at_level(logging.WARNING, logger="phosp"), \
         _patched_gmx(), \
         patch.object(p_new, "_build_stage", return_value=mock_stage):
        p_new.execute(only_stages="1")  # stage1 already complete — loop skips it

    assert any("Config has changed" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run the test to confirm it fails**

```
pytest tests/test_pipeline.py::test_config_hash_guard_warns_on_mismatch -v
```
Expected: FAIL — `mark_complete` doesn't accept `config_hash`, `Pipeline` doesn't accept `config_path`.

- [ ] **Step 3: Update `Checkpoint` in `phosp/utils/checkpoint.py`**

```python
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path


class Checkpoint:
    def __init__(self, checkpoint_file: Path) -> None:
        self.path = checkpoint_file
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {"completed_stages": [], "artifacts": {}}

    def mark_complete(
        self,
        stage: str,
        artifacts: dict[str, str],
        config_hash: str | None = None,
    ) -> None:
        if stage not in self._data["completed_stages"]:
            self._data["completed_stages"].append(stage)
        self._data["artifacts"][stage] = artifacts
        self._data[f"{stage}_completed_at"] = datetime.now().isoformat()
        if config_hash is not None and "config_hash" not in self._data:
            self._data["config_hash"] = config_hash
        self.path.write_text(json.dumps(self._data, indent=2))

    def is_complete(self, stage: str) -> bool:
        return stage in self._data["completed_stages"]

    def get_artifacts(self, stage: str) -> dict[str, str]:
        return self._data["artifacts"].get(stage, {})

    def get_config_hash(self) -> str | None:
        return self._data.get("config_hash")

    def store_config_hash(self, hash_value: str) -> None:
        if "config_hash" not in self._data:
            self._data["config_hash"] = hash_value
            self.path.write_text(json.dumps(self._data, indent=2))
```

- [ ] **Step 4: Update `Pipeline.__init__` and `execute` in `phosp/pipeline.py`**

Add `import hashlib` at the top of the file.

Replace `Pipeline.__init__`:

```python
def __init__(
    self,
    config: PhospConfig,
    output_root: Path,
    ui: PhospUI | None = None,
    config_path: Path | None = None,
) -> None:
    self.config = config
    self.output_root = output_root
    self.ui = ui
    self.output_root.mkdir(parents=True, exist_ok=True)
    self.checkpoint = Checkpoint(output_root / "checkpoint.json")
    self._config_hash: str | None = None
    if config_path is not None:
        try:
            self._config_hash = hashlib.sha256(
                config_path.read_bytes()
            ).hexdigest()[:16]
        except OSError:
            pass
```

In `execute`, after `self._preflight_checks()` and before `self._clean_orphan_tmpdirs()`, add:

```python
        if self._config_hash:
            stored = self.checkpoint.get_config_hash()
            if stored and stored != self._config_hash:
                logger.warning(
                    "Config has changed since this run was started. "
                    "Completed stages used the previous config. "
                    "Use --start-from stage1 to re-run from scratch."
                )
```

In `_run_stage`, replace the `self.checkpoint.mark_complete(stage_name, remapped)` call with:

```python
            self.checkpoint.mark_complete(stage_name, remapped, config_hash=self._config_hash)
```

- [ ] **Step 5: Pass `config_path` from CLI `run` command in `phosp/cli.py`**

Replace the two `Pipeline(...)` calls in the `run` function:

Dry-run path (line ~121):
```python
        p = Pipeline(cfg, output_root=output_root, config_path=config_path)
```

Normal path (line ~130):
```python
    Pipeline(cfg, output_root=config_path.parent / "output", ui=ui, config_path=config_path).execute(
        start_from=start_from, only_stages=stages
    )
```

- [ ] **Step 6: Run all tests**

```
pytest tests/ -v --tb=short
```
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add phosp/utils/checkpoint.py phosp/pipeline.py phosp/cli.py tests/test_pipeline.py
git commit -m "feat: config hash guard warns when config changes between pipeline runs"
```

---

### Task 6: Elapsed time in `phosp status`

**Files:**
- Modify: `phosp/utils/checkpoint.py` — add `mark_stage_started`, update `mark_complete` for duration, add `get_duration`
- Modify: `phosp/pipeline.py` — call `mark_stage_started` before each stage
- Modify: `phosp/cli.py` — add "Duration" column to status table
- Modify: `tests/test_cli.py` — add test for Duration column

**Interfaces:**
- `Checkpoint.mark_stage_started(stage: str)` — writes `{stage}_started_at`
- `Checkpoint.get_duration(stage: str) -> float | None` — returns `{stage}_duration_seconds`
- `mark_complete` computes and stores `{stage}_duration_seconds` from `{stage}_started_at`

- [ ] **Step 1: Write the failing tests**

```python
# In tests/test_cli.py, add:

def test_status_shows_duration_column(tmp_path: Path):
    """Status table includes a Duration column with formatted elapsed time."""
    out = tmp_path / "output"
    out.mkdir()
    (out / "checkpoint.json").write_text(json.dumps({
        "completed_stages": ["stage1"],
        "artifacts": {},
        "stage1_completed_at": "2026-06-30T10:00:42",
        "stage1_duration_seconds": 42.0,
    }))
    result = runner.invoke(app, ["status", str(out)])
    assert "Duration" in result.output
    assert "42 s" in result.output


def test_checkpoint_stores_duration_after_stage_completes(tmp_path: Path):
    """mark_stage_started + mark_complete stores duration_seconds."""
    from phosp.utils.checkpoint import Checkpoint
    cp = Checkpoint(tmp_path / "checkpoint.json")
    cp.mark_stage_started("stage1")
    cp.mark_complete("stage1", {})
    dur = cp.get_duration("stage1")
    assert dur is not None
    assert dur >= 0.0
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_cli.py::test_status_shows_duration_column \
       tests/test_cli.py::test_checkpoint_stores_duration_after_stage_completes -v
```
Expected: FAIL — `Checkpoint` has no `mark_stage_started` or `get_duration`.

- [ ] **Step 3: Update `Checkpoint` in `phosp/utils/checkpoint.py`**

Add to the class:

```python
    def mark_stage_started(self, stage: str) -> None:
        self._data[f"{stage}_started_at"] = datetime.now().isoformat()
        self.path.write_text(json.dumps(self._data, indent=2))

    def get_duration(self, stage: str) -> float | None:
        return self._data.get(f"{stage}_duration_seconds")
```

Update `mark_complete` to compute duration. After the line `self._data[f"{stage}_completed_at"] = datetime.now().isoformat()`, add:

```python
        started_at_str = self._data.get(f"{stage}_started_at")
        if started_at_str:
            try:
                started = datetime.fromisoformat(started_at_str)
                duration = (datetime.now() - started).total_seconds()
                self._data[f"{stage}_duration_seconds"] = duration
            except (ValueError, TypeError):
                pass
```

- [ ] **Step 4: Call `mark_stage_started` in `Pipeline._run_stage`**

In `phosp/pipeline.py`, at the start of the `try:` block in `_run_stage` (before `stage.validate_inputs()`), add:

```python
        try:
            self.checkpoint.mark_stage_started(stage_name)
            stage.validate_inputs()
            result = stage.run()
            ...
```

- [ ] **Step 5: Add "Duration" column to `status` in `phosp/cli.py`**

Add a helper function before the `status` command function:

```python
def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 1:
        return "< 1 s"
    if seconds < 60:
        return f"{int(seconds)} s"
    mins, secs = divmod(int(seconds), 60)
    if mins < 60:
        return f"{mins} m {secs} s"
    hours, mins = divmod(mins, 60)
    return f"{hours} h {mins} m"
```

In the `status` function, add a "Duration" column to the table:

```python
    table.add_column("Duration")
```

(after the "Key Artifacts" column add)

And update `table.add_row(...)` to include the duration value:

```python
        if s in completed:
            status_str = "[green]✓ complete[/]"
            completed_at = data.get(f"{s}_completed_at", "")
            artifacts = data.get("artifacts", {}).get(s, {})
            artifact_str = ", ".join(Path(v).name for v in list(artifacts.values())[:3])
            duration_str = _fmt_duration(data.get(f"{s}_duration_seconds"))
        else:
            status_str = "[dim]pending[/]"
            completed_at = ""
            artifact_str = ""
            duration_str = "—"
        table.add_row(_labels[s], status_str, completed_at, artifact_str, duration_str)
```

- [ ] **Step 6: Run all tests**

```
pytest tests/ -v --tb=short
```
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add phosp/utils/checkpoint.py phosp/pipeline.py phosp/cli.py \
        tests/test_cli.py
git commit -m "feat: elapsed time tracking and Duration column in phosp status"
```

---

### Task 7: GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Triggers on push and pull_request to `master`
- Matrix: Python 3.10, 3.11, 3.12
- No GROMACS or external tools needed — all external calls mocked in tests

- [ ] **Step 1: Create `.github/workflows/ci.yml`**

```bash
mkdir -p .github/workflows
```

Write `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [master]
  pull_request:
    branches: [master]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run tests
        run: pytest --tb=short
```

- [ ] **Step 2: Verify the YAML is syntactically valid**

```
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"
```
Expected: no output (no error).

- [ ] **Step 3: Verify tests still pass locally**

```
pytest --tb=short
```
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "feat: GitHub Actions CI on Python 3.10/3.11/3.12"
```

---

## Self-review

**Spec coverage:**
- Task 1 → spec item 5 (--stages/--start-from upfront validation) ✓
- Task 2 → spec item 4 (unknown plugin names → error) ✓
- Task 3 → spec item 1 (subprocess timeout) ✓
- Task 4 → spec item 2 (configurable pdb2pqr) ✓
- Task 5 → spec item 3 (config hash guard) ✓
- Task 6 → spec item 7 (phosp status elapsed time) ✓
- Task 7 → spec item 6 (GitHub Actions CI) ✓

**Type consistency:**
- `mark_complete(stage, artifacts, config_hash=None)` — used with this exact signature in Task 5 test and `_run_stage`
- `mark_stage_started(stage)` — called in `_run_stage` and in Task 6 test
- `get_duration(stage)` — called in CLI status and in Task 6 test
- `GROMACSEngine(binary=..., timeout_minutes=...)` — matches test in Task 3
- `protonate_structure(..., pdb2pqr_binary=...)` — matches stage1 call and test in Task 4

**Placeholder scan:** No TBDs, no "implement later", no "similar to Task N" without full code.

**Backward compatibility:** All new parameters are optional with defaults matching current behavior. Existing checkpoint files without new fields load fine (`dict.get` returns None). `mark_complete(stage, artifacts)` without `config_hash` still works.
