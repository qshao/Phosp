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


def test_status_shows_duration_column(tmp_path: Path):
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
    from phosp.utils.checkpoint import Checkpoint
    cp = Checkpoint(tmp_path / "checkpoint.json")
    cp.mark_stage_started("stage1")
    cp.mark_complete("stage1", {})
    dur = cp.get_duration("stage1")
    assert dur is not None
    assert dur >= 0.0


def test_clean_removes_output_dir(tmp_path: Path):
    """phosp clean deletes the output directory after confirmation."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "checkpoint.json").write_text('{"completed_stages": []}')
    (output_dir / "stage1").mkdir()

    result = runner.invoke(app, ["clean", str(output_dir)], input="y\n")
    assert result.exit_code == 0, result.output
    assert not output_dir.exists()


def test_clean_aborts_on_no(tmp_path: Path):
    """phosp clean aborts without deleting when user says no."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = runner.invoke(app, ["clean", str(output_dir)], input="n\n")
    assert result.exit_code != 0
    assert output_dir.exists()


def test_clean_rejects_file_path(tmp_path: Path):
    """phosp clean exits cleanly when given a file path instead of directory."""
    f = tmp_path / "checkpoint.json"
    f.write_text("{}")
    result = runner.invoke(app, ["clean", str(f)])
    assert result.exit_code == 1
    assert f.exists()


def test_clean_missing_dir(tmp_path: Path):
    """phosp clean exits with error if output_dir doesn't exist."""
    result = runner.invoke(app, ["clean", str(tmp_path / "nonexistent")])
    assert result.exit_code == 1


def test_run_reference_dry_run_uses_reference_output_dir(tmp_path: Path):
    """phosp run --reference --dry-run creates output_reference/ not output/."""
    import shutil as _shutil
    from unittest.mock import patch, MagicMock

    config_path = tmp_path / "config.yaml"
    _shutil.copy(
        Path(__file__).parent / "fixtures" / "valid_config.yaml",
        config_path,
    )
    # Patch the PDB path in the config so load_config doesn't fail
    cfg_text = config_path.read_text().replace(
        "tests/fixtures/ubiquitin.pdb",
        str(Path(__file__).parent / "fixtures" / "ubiquitin.pdb"),
    )
    config_path.write_text(cfg_text)

    mock_pipeline = MagicMock()
    mock_pipeline._preflight_checks.return_value = None
    mock_pipeline.execute.return_value = None

    with patch("phosp.pipeline.Pipeline", return_value=mock_pipeline) as MockPipeline:
        result = runner.invoke(app, ["run", str(config_path), "--reference", "--dry-run"])

    assert MockPipeline.called, f"Pipeline not called. Output: {result.output}"
    call_kwargs = MockPipeline.call_args
    # output_root is the second positional arg or keyword
    output_root = call_kwargs.kwargs.get("output_root") or call_kwargs.args[1]
    assert "reference" in str(output_root), (
        f"Expected 'reference' in output_root when --reference is passed, got: {output_root}"
    )
    assert call_kwargs.kwargs.get("reference_mode") is True, (
        f"Expected reference_mode=True, got: {call_kwargs.kwargs}"
    )


def test_run_reference_non_dry_run_passes_reference_mode(tmp_path: Path):
    """--reference passes reference_mode=True and output_reference/ to Pipeline in normal run."""
    import shutil as _shutil
    from unittest.mock import patch, MagicMock

    config_path = tmp_path / "config.yaml"
    _shutil.copy(Path(__file__).parent / "fixtures" / "valid_config.yaml", config_path)
    cfg_text = config_path.read_text().replace(
        "tests/fixtures/ubiquitin.pdb",
        str(Path(__file__).parent / "fixtures" / "ubiquitin.pdb"),
    )
    config_path.write_text(cfg_text)

    mock_pipeline = MagicMock()
    mock_pipeline.execute.return_value = None

    with patch("phosp.pipeline.Pipeline", return_value=mock_pipeline) as MockPipeline:
        result = runner.invoke(app, ["run", str(config_path), "--reference"])

    assert MockPipeline.called, f"Pipeline not called. Output: {result.output}"
    kwargs = MockPipeline.call_args.kwargs
    output_root = kwargs.get("output_root") or MockPipeline.call_args.args[1]
    assert "reference" in str(output_root), f"Expected 'reference' in output_root, got: {output_root}"
    assert kwargs.get("reference_mode") is True, f"Expected reference_mode=True, got: {kwargs}"
