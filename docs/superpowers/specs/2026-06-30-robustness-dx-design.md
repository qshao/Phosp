# Design: robustness + DX improvements

**Date:** 2026-06-30  
**Status:** approved  
**Scope:** 7 independent improvements across safety/correctness and developer experience

---

## Items

### 1. Subprocess timeout

**Problem:** `subprocess.run` calls in `phosp/engines/gromacs.py` have no timeout. A hung `mdrun` (OOM, GPU stall, NFS hang) blocks the process forever.

**Config change:**
```yaml
gromacs:
  binary: gmx
  timeout_minutes: ~      # null = no limit (default); set to e.g. 120 for a 2-hour limit
```

Add `timeout_minutes: int | None = None` to `GROMACSConfig` in `phosp/config.py`.

**Implementation:**
- `_run_gmx(cmd, cwd, input_text="", timeout=None)` — pass `timeout` (in seconds, converted from minutes) to `subprocess.run`.
- `GROMACSEngine.__init__` gains `timeout_minutes: int | None = None`; stored as `self._timeout`.
- All internal `_run_gmx` calls in `GROMACSEngine` pass `self._timeout * 60` when not None.
- `pipeline._run_stage` passes `binary=…, timeout_minutes=…` when constructing `GROMACSEngine`.
- On `subprocess.TimeoutExpired`, catch and re-raise as `SimulationError(f"GROMACS timed out after {timeout_minutes} minutes: {cmd[0]} {cmd[1]}")`.

**Tests:** mock `subprocess.run` to raise `TimeoutExpired`; assert `SimulationError` with "timed out" in message.

---

### 2. Configurable pdb2pqr binary

**Problem:** `protonate_structure` in `phosp/utils/structure.py` hardcodes `"pdb2pqr"`. Users with pdb2pqr in a virtualenv or a custom path cannot override it.

**Config change:**
```yaml
gromacs:
  binary: gmx
  pdb2pqr: pdb2pqr        # name or full path, e.g. "/opt/pdb2pqr/bin/pdb2pqr"
```

Add `pdb2pqr: str = "pdb2pqr"` to `GROMACSConfig`.

**Implementation:**
- `protonate_structure(pdb, output, ph, pdb2pqr_binary="pdb2pqr")` — gains a `pdb2pqr_binary` kwarg.
- `Stage2Prepare.run` reads `self.config.gromacs.pdb2pqr` and passes it through.
- `phosp validate` and `pipeline._preflight_checks` check `shutil.which(cfg.gromacs.pdb2pqr)` (same pattern as `gromacs.binary` check).
- Starter config (`phosp init`) and README document the field.

**Tests:** assert `protonate_structure` uses the configured binary in the subprocess command.

---

### 3. Config hash guard in checkpoint

**Problem:** If a user changes `modification.sites` or `simulation.production_time_ns` and re-runs, phosp silently resumes using the old completed stages, producing a mixed-config result.

**Implementation:**
- `load_config` returns `(PhospConfig, raw_yaml_bytes: bytes)` — or expose a helper `config_hash(path) -> str` that reads and hashes. Simpler: hash in `Pipeline.__init__` by re-reading the config file.
- `Pipeline.__init__` gains an optional `config_path: Path | None = None` parameter. The CLI passes it. The hash is computed as `hashlib.sha256(config_path.read_bytes()).hexdigest()[:16]` when available; skipped when None (e.g. in tests that construct Pipeline directly).
- `Checkpoint.mark_complete` also writes `config_hash` on the first complete stage if not already stored.
- `Pipeline.execute` (before the stage loop): if `checkpoint.get("config_hash")` exists and differs, log a `WARNING`: `"Config has changed since this run was started. Stages already completed used the previous config. Use --start-from stage1 to re-run from scratch."` Do not abort.
- `Checkpoint` gains `store_config_hash(hash: str)` and `get_config_hash() -> str | None`.

**Tests:** write a config, run one stage, change the config file, assert the warning is logged on the next execute call.

---

### 4. Unknown plugin names → error

**Problem:** A typo in `analysis.plugins` (e.g. `rmsdf`) is silently skipped with a `logger.warning`. The user gets a partial report with no obvious error.

**Implementation:**  
In `Stage4Analyze.run`, before the plugin loop:

```python
registry = _discover_plugins()
unknown = [p for p in cfg.analysis.plugins if p not in registry]
if unknown:
    raise AnalysisError(
        f"Unknown analysis plugins: {unknown}. "
        f"Valid plugins: {sorted(registry)}"
    )
```

This raises before any analysis runs, so the user sees the typo immediately.

**Tests:** assert `AnalysisError` is raised when `analysis.plugins` contains an unknown name; assert valid names are listed in the message.

---

### 5. `--stages` / `--start-from` upfront validation

**Problem:** `pipeline._resolve_stages` builds `["stageX"]` from user input and only fails at `_build_stage`'s `case _: raise` branch, which is mid-run after preflight.

**Implementation:**  
In `_resolve_stages`, after building the list, validate:

```python
invalid = [s for s in stages if s not in _ALL_STAGES]
if invalid:
    raise PhospError(
        f"Unknown stage(s): {invalid}. "
        f"Valid stages: {_ALL_STAGES}"
    )
```

Same check applies to `start_from`: if provided and not in `_ALL_STAGES`, raise immediately.

**Tests:** assert `PhospError` with "Unknown stage" when `--stages` contains invalid input; assert it fires before any stage runs.

---

### 6. GitHub Actions CI

**File:** `.github/workflows/ci.yml`

**Trigger:** `push` and `pull_request` on `master`.

**Matrix:** Python 3.10, 3.11, 3.12.

**Steps:**
1. `actions/checkout@v4`
2. `actions/setup-python@v5` with matrix Python version
3. `pip install -e ".[dev]"`
4. `pytest --tb=short`

No GROMACS or pdb2pqr required — all external calls are mocked in the test suite. Estimated runtime ~3 min per matrix leg.

---

### 7. `phosp status` — elapsed time

**Problem:** The checkpoint stores `stage_completed_at` but not when the stage started, so duration cannot be computed. The status table has no duration column.

**Checkpoint changes:**
- `Pipeline._run_stage` writes `checkpoint.mark_stage_started(stage_name)` before `stage.validate_inputs()` runs.
- `Checkpoint.mark_stage_started(stage)` stores `{stage}_started_at = datetime.now().isoformat()`.
- `Checkpoint.mark_complete` computes and stores `{stage}_duration_seconds` from start→complete timestamps.
- `Checkpoint.get_duration(stage) -> float | None` returns the stored duration.

**Status display:**
- Add a "Duration" column to the `phosp status` table.
- Format: `< 1 s`, `42 s`, `14 m 22 s`, `2 h 14 m`.
- Stages with no duration (pending, or started before this feature) show `—`.

**Tests:** assert duration is stored in checkpoint after a stage completes; assert formatted display includes the Duration column.

---

## Implementation order

1. Stage name validation (trivial, no deps)
2. Unknown plugin error (trivial, no deps)
3. Subprocess timeout (self-contained engine change)
4. pdb2pqr configurable (config + stage2 + preflight)
5. Config hash guard (checkpoint + pipeline)
6. Status elapsed time (checkpoint + CLI)
7. GitHub Actions CI (new file, no code deps)
