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


def test_amber_ff14sb_blocked_at_preflight(tmp_path):
    """amber_ff14sb raises PhospError at preflight, before any stage runs."""
    cfg = load_config(FIXTURES / "valid_config.yaml")
    cfg.forcefield = "amber_ff14sb"
    p = Pipeline(cfg, output_root=tmp_path / "output")
    with _patched_gmx():
        with pytest.raises(PhospError, match="AMBER ff14SB"):
            p.execute(only_stages="1")


def test_stage_failure_preserves_previous_final_dir(tmp_path):
    """When stage.run() raises, a previously completed final_dir is not deleted."""
    p = _make_pipeline(tmp_path)
    final_dir = tmp_path / "output" / "stage1"
    final_dir.mkdir(parents=True)
    keeper = final_dir / "precious.txt"
    keeper.write_text("previous good result")

    mock_stage = MagicMock()
    mock_stage.run.side_effect = RuntimeError("mid-run failure")

    with _patched_gmx(), patch.object(p, "_build_stage", return_value=mock_stage):
        with pytest.raises(RuntimeError, match="mid-run failure"):
            p.execute(only_stages="1")

    assert final_dir.exists(), "previous final_dir must survive a failed re-run"
    assert keeper.read_text() == "previous good result"


def test_disk_warning_logged_when_low(tmp_path, caplog):
    import logging
    cfg = load_config(FIXTURES / "valid_config.yaml")
    cfg.simulation.production_time_ns = 100.0
    p = Pipeline(cfg, output_root=tmp_path / "output")

    mock_stage = MagicMock()
    mock_stage.run.return_value = MagicMock(artifacts={})

    with _patched_gmx(), \
         patch("phosp.pipeline.shutil.disk_usage",
               return_value=MagicMock(free=1 * 1024 ** 3)), \
         patch.object(p, "_build_stage", return_value=mock_stage), \
         caplog.at_level(logging.WARNING, logger="phosp"):
        p.execute(only_stages="1")

    assert any("disk space" in r.message.lower() for r in caplog.records)
