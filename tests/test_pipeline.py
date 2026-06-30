from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from phosp.config import load_config
from phosp.pipeline import Pipeline

FIXTURES = Path(__file__).parent / "fixtures"


def _make_pipeline(tmp_path):
    cfg = load_config(FIXTURES / "valid_config.yaml")
    return Pipeline(cfg, output_root=tmp_path / "output")


def test_pipeline_creates_output_dir(tmp_path):
    p = _make_pipeline(tmp_path)
    assert (tmp_path / "output").exists()


def test_pipeline_skips_completed_stages(tmp_path):
    p = _make_pipeline(tmp_path)
    p.checkpoint.mark_complete("stage1", {"modified_pdb": "fake.pdb"})
    # stage1 run() should not be called
    mock_stage = MagicMock()
    with patch.object(p, "_build_stage", return_value=mock_stage):
        p.execute(only_stages="1")
    mock_stage.run.assert_not_called()


def test_start_from_skips_earlier_stages(tmp_path):
    p = _make_pipeline(tmp_path)
    called = []
    p._run_stage = lambda name, *a, **kw: called.append(name)
    p.execute(start_from="stage3", only_stages="1,2,3")
    assert "stage1" not in called
    assert "stage2" not in called
