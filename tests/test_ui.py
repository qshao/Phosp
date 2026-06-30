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
