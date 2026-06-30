from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import patch
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
    assert result.exit_code == 0, result.output
    assert "valid" in result.output.lower()


def test_run_dry_run_skips_execution(tmp_path: Path):
    """Test that dry-run runs preflight checks but does not execute stages."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text((FIXTURES / "valid_config.yaml").read_text())
    out = tmp_path / "output"

    # Patch which to simulate tools present; patch _check_forcefield to skip FF check
    with (
        patch("phosp.pipeline.shutil.which", return_value="/usr/bin/gmx"),
        patch("shutil.which", return_value="/usr/bin/gmx"),
        patch("phosp.pipeline.Pipeline._check_forcefield"),
    ):
        result = runner.invoke(app, ["run", str(config_path), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "complete" in result.output.lower()

    # No stage directories created
    if out.exists():
        for stage_num in range(1, 5):
            assert not (out / f"stage{stage_num}").exists()
