from pathlib import Path
from unittest.mock import patch, MagicMock
import json
import pytest
from phosp.config import load_config
from phosp.stages.stage2_prepare import Stage2Prepare
from phosp.exceptions import StageInputError

FIXTURES = Path(__file__).parent / "fixtures"


def _make_stage(tmp_path):
    cfg = load_config(FIXTURES / "valid_config.yaml")
    cfg.input.path = FIXTURES / "ubiquitin.pdb"
    stage1_dir = tmp_path / "output" / "stage1"
    stage1_dir.mkdir(parents=True)
    import shutil
    shutil.copy(FIXTURES / "ubiquitin.pdb", stage1_dir / "modified.pdb")
    manifest = [{"chain": "A", "resid": 66, "original_resname": "THR",
                 "mod_type": "pThr", "new_resname": "TPO"}]
    (stage1_dir / "modification_manifest.json").write_text(json.dumps(manifest))
    engine = MagicMock()
    engine.prepare_topology.return_value = tmp_path / "output" / "stage2" / "topol.top"
    engine.solvate.return_value = (tmp_path / "output" / "stage2" / "solvated.gro",
                                   tmp_path / "output" / "stage2" / "topol.top")
    engine.add_ions.return_value = (tmp_path / "output" / "stage2" / "ions.gro",
                                    tmp_path / "output" / "stage2" / "topol.top")
    engine.generate_mdp.return_value = tmp_path / "output" / "stage2" / "minimization.mdp"
    ff = MagicMock()
    ff.patch_topology.return_value = tmp_path / "output" / "stage2" / "topol.top"
    return Stage2Prepare(cfg, engine, ff, tmp_path / "output" / "stage2")


def test_stage2_validate_raises_if_stage1_missing(tmp_path):
    cfg = load_config(FIXTURES / "valid_config.yaml")
    cfg.input.path = FIXTURES / "ubiquitin.pdb"
    stage = Stage2Prepare(cfg, MagicMock(), MagicMock(), tmp_path / "stage2")
    with pytest.raises(StageInputError, match="modified.pdb"):
        stage.validate_inputs()


def test_stage2_run_calls_engine_methods(tmp_path):
    stage = _make_stage(tmp_path)
    # Create fake output files so stage completes
    out = tmp_path / "output" / "stage2"
    out.mkdir(parents=True, exist_ok=True)
    for f in ["topol.top", "solvated.gro", "ions.gro",
              "minimization.mdp", "nvt.mdp", "npt.mdp", "production.mdp"]:
        (out / f).write_text(f"; fake {f}")
    result = stage.run()
    assert result.stage == "stage2"
    stage.engine.prepare_topology.assert_called_once()
    stage.engine.solvate.assert_called_once()
    stage.engine.add_ions.assert_called_once()
