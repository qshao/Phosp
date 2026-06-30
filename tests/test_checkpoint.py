import json
from pathlib import Path
import pytest
from phosp.utils.checkpoint import Checkpoint

def test_mark_and_query_complete(tmp_path):
    cp = Checkpoint(tmp_path / "checkpoint.json")
    assert not cp.is_complete("stage1")
    cp.mark_complete("stage1", {"modified_pdb": "output/stage1/modified.pdb"})
    assert cp.is_complete("stage1")

def test_artifacts_round_trip(tmp_path):
    cp = Checkpoint(tmp_path / "checkpoint.json")
    cp.mark_complete("stage1", {"key": "value"})
    cp2 = Checkpoint(tmp_path / "checkpoint.json")
    assert cp2.get_artifacts("stage1") == {"key": "value"}

def test_missing_stage_not_complete(tmp_path):
    cp = Checkpoint(tmp_path / "checkpoint.json")
    assert not cp.is_complete("stage2")
    assert cp.get_artifacts("stage2") == {}
