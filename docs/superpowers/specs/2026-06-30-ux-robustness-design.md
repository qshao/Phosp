# phosp UX & Robustness Design

**Date:** 2026-06-30  
**Branch base:** master (commit 4063561)  
**Scope:** User-friendliness and robustness improvements across three pillars — atomic stage writes, Rich terminal UI, and pre-flight guardrails.

---

## 1. Atomic Stage Writes

### Problem

If a run is interrupted mid-stage (Ctrl+C, crash, SIGKILL), partial outputs accumulate in the stage output directory. On the next run, the stage is not in the checkpoint so it reruns, but it finds a directory with stale partial files — undefined behavior depending on which GROMACS commands happened to complete.

### Design

Pipeline manages a sibling temp directory (`.stageN_tmp`) at the same level as the final output directory. The stage is constructed with `output_root=tmp_dir`, so all writes go there. On success, `tmp_dir.rename(final_dir)` completes the stage atomically (POSIX rename is atomic when source and destination are on the same filesystem). On any failure or interrupt, a `try/finally` block deletes the temp dir entirely.

**Temp dir naming:** `output_root / f".{stage_name}_tmp"` — e.g. `.stage2_tmp` alongside `stage2/`.

**`validate_inputs()` compatibility:** Stages navigate to previous stages via `self.output_root.parent / "stageN"`. Because the temp dir is a sibling of the final dirs (same parent), these paths resolve correctly without any changes to stage code.

**Orphan cleanup:** At startup, `Pipeline.execute()` scans `output_root` for dirs matching `.stage*_tmp` and deletes them before running. This handles the SIGKILL case where `finally` blocks did not execute.

### What Changes

| File | Change |
|------|--------|
| `phosp/pipeline.py` | `_run_stage()` creates tmp dir, wraps stage execution in try/finally, renames on success. `execute()` adds orphan cleanup scan. |
| Stage files | No changes. |

---

## 2. Rich Terminal UI

### Problem

`phosp run` is completely silent — output only appears if the caller configures Python logging. Users running long simulations have no visibility into progress, elapsed time, or where a failure occurred.

### Design

**`phosp/ui.py` — `PhospUI` class:**

A thin wrapper around `rich.console.Console` and `rich.progress.Progress` (spinner style). Three lifecycle methods:

- `stage_start(name: str, description: str)` — prints bold stage banner, starts live spinner
- `stage_complete(name: str, elapsed_s: float)` — stops spinner, prints green `✓` line with elapsed time
- `stage_error(name: str, exc: Exception)` — stops spinner, renders a red `rich.panel.Panel` with exception type and message (not a raw traceback)

`Pipeline.__init__` accepts `ui: PhospUI | None = None`. When `None` (library use), all hooks are no-ops. The CLI passes a `PhospUI()` instance.

**Example output:**
```
▶  Stage 1 — Chemical Modification
   ⠸ Protonating structure...
✓  Stage 1 complete  (12s)

▶  Stage 2 — MD Preparation
   ⠸ Running pdb2gmx...
✓  Stage 2 complete  (1m 04s)
```

**`phosp status <output_dir>` CLI command:**

Reads `checkpoint.json` and renders a `rich.table.Table` with columns: Stage, Status, Completed At, Key Artifacts. Shows all four stages regardless of whether they've run. Exits with code 0 if all four stages are marked complete, 1 otherwise — so `phosp status` returning 0 is a reliable CI gate, but a partial run (only stages 1-2 done) always returns 1.

**Stage 4 plugin progress:**

`Stage4Analyze.run()` calls `ui.stage_start()` per-plugin when a `ui` is provided, so users see which analysis plugin is running.

**`ui` propagation to stages:** `Stage.__init__` gains an optional `ui: PhospUI | None = None` parameter (added to the base class). `Pipeline._build_stage()` passes `ui=self.ui` when constructing each stage. This is the only change to the Stage base class interface.

### What Changes

| File | Change |
|------|--------|
| `phosp/ui.py` | New file — `PhospUI` class |
| `phosp/pipeline.py` | Accept `ui` param; call hooks around `_run_stage` |
| `phosp/cli.py` | Create `PhospUI()`, pass to Pipeline; add `status` command |
| `phosp/stages/base.py` | Add `ui: PhospUI | None = None` to `Stage.__init__` |
| `phosp/stages/stage4_analyze.py` | Use `self.ui`; call per-plugin hooks |

---

## 3. Guardrails

### 3a. Dependency Check

`Pipeline.execute()` calls `shutil.which("gmx")` before the stage loop. If missing, raises `PhospError("GROMACS (gmx) not found in PATH. Install GROMACS and ensure gmx is on your PATH.")` immediately — before any output directory is created.

### 3b. Stricter Config Validation

Additional validators in `config.py`:

| Field | Validator type | Rule |
|-------|---------------|------|
| `production_time_ns` | `@field_validator` | `> 0` |
| `output_freq_ps` | `@field_validator` | `> 0` |
| `output_freq_ps` vs `production_time_ns` | `@model_validator(mode="after")` on `SimulationConfig` | `output_freq_ps <= production_time_ns * 1000` (at least one output frame) |
| `salt_concentration_mM` | `@field_validator` | `>= 0` |
| `HPCConfig.ntasks` | `@field_validator` | `>= 1` |
| `HPCConfig.gpus` | `@field_validator` | `>= 0` |

All errors surface at `load_config()` time with clean Pydantic messages.

### 3c. Disk Space Warning

After the dependency check, `Pipeline.execute()` estimates required space:

```
estimated_gb = production_time_ns * 1.0 + 0.5   # 1 GB/ns + 500 MB fixed
```

Uses `shutil.disk_usage(output_root)` to get available space. If available < estimated, logs a `WARNING` with the estimate and available space. Never blocks execution — users on HPC with quota limits get a heads-up.

### 3d. `phosp init` Command

`phosp init [path]` (default: `./phosp_config.yaml`) writes a fully-commented starter config YAML covering every key with valid values and hints. Includes a footer comment: `# Next: phosp predict-sites phosp_config.yaml`. If the file already exists, prints `Error: phosp_config.yaml already exists. Use a different path or delete it first.` and exits non-zero without overwriting.

### 3e. `--dry-run` Flag

`phosp run config.yaml --dry-run`:
1. Loads and validates config
2. Checks input PDB exists (if `source=pdb`)
3. Checks `gmx` in PATH
4. Prints disk space estimate
5. Prints `Dry run complete — no stages executed`
6. Exits 0

Never creates output dirs, never runs subprocesses.

### What Changes

| File | Change |
|------|--------|
| `phosp/config.py` | Add field validators |
| `phosp/pipeline.py` | Add dependency check + disk warning to `execute()` |
| `phosp/cli.py` | Add `init` command; add `--dry-run` to `run` |

---

## 4. Resilience

### 4a. Plugin Fail-Soft (Stage 4)

Each plugin in `Stage4Analyze.run()` runs inside its own `try/except Exception`. Failures are collected as `list[tuple[str, str]]` (plugin name, error message). Analysis continues to the next plugin.

After all plugins run:
- If any failures: log each as `WARNING`; include a warning banner per failed plugin in the HTML report
- If **all** requested plugins failed: raise `AnalysisError("All analysis plugins failed")` with the collected errors
- If at least one succeeded: return normally (partial results are valid)

The HTML report template gains a `{% if failed_plugins %}` warning section listing each plugin name and its error.

### 4b. Logging Configuration

New `phosp/logging.py`:

```python
def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
```

- Adds one `StreamHandler` (stderr) with format `%(asctime)s [%(levelname)s] %(name)s: %(message)s`
- Optional `FileHandler` when `log_file` is provided
- Attaches to the `phosp` root logger — all submodule loggers inherit it
- Library users who do not call this see no output (standard Python convention — no default handler)

CLI calls `configure_logging()` at startup. `phosp run` gains `--log-level` (default `INFO`) and `--log-file` options.

### What Changes

| File | Change |
|------|--------|
| `phosp/stages/stage4_analyze.py` | Per-plugin try/except; collect failures; pass to report renderer |
| `phosp/templates/report.html.j2` | Add `{% if failed_plugins %}` warning banner |
| `phosp/logging.py` | New file — `configure_logging()` |
| `phosp/cli.py` | Call `configure_logging()` at startup; add `--log-level` / `--log-file` |
| `phosp/pipeline.py` | Orphan temp dir cleanup at start of `execute()` (see §1) |

---

## Global Constraints

- Python ≥ 3.10; `from __future__ import annotations`
- No `print()` in library code — use `logging.getLogger(__name__)`
- All file I/O via `pathlib.Path`
- Pydantic v2 (`model_validate`, `@field_validator`, `model_config`)
- `rich` is already a transitive dependency of Typer — no new top-level dependency added
- No comments unless WHY is non-obvious

## Files Added / Modified Summary

| File | Action |
|------|--------|
| `phosp/ui.py` | **New** |
| `phosp/logging.py` | **New** |
| `phosp/pipeline.py` | Modified |
| `phosp/cli.py` | Modified |
| `phosp/config.py` | Modified |
| `phosp/stages/stage4_analyze.py` | Modified |
| `phosp/templates/report.html.j2` | Modified |

## Test Coverage Expected

- `tests/test_pipeline.py` — atomic write (success renames, failure cleans up, orphan removed), dependency check, disk warning, dry-run
- `tests/test_cli.py` — `init` creates file, `status` renders table and correct exit code
- `tests/test_config.py` — new field validator cases
- `tests/test_stage4.py` — plugin fail-soft (partial success, all-fail raises)
- `tests/test_logging.py` — `configure_logging` attaches handler to phosp logger
