import json
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from phosp.config import load_config
from phosp.pipeline import Pipeline

FIXTURES = Path(__file__).parent / "fixtures"


def _mock_engine():
    engine = MagicMock()

    def _prepare_topology(*a, **kw):
        out_dir = kw.get("output_dir")
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "topol.top"
        p.write_text("; fake topology")
        return p

    def _solvate(gro, top, **kw):
        p = top.parent / "solvated.gro"
        p.write_text("; fake")
        return p, top

    def _add_ions(gro, top, **kw):
        p = top.parent / "ions.gro"
        p.write_text("; fake ions")
        return p, top

    def _generate_mdp(phase, proto, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{phase}.mdp"
        p.write_text(f"; {phase}")
        return p

    engine.prepare_topology.side_effect = _prepare_topology
    engine.solvate.side_effect = _solvate
    engine.add_ions.side_effect = _add_ions
    engine.generate_mdp.side_effect = _generate_mdp
    return engine


def test_pipeline_stages_1_and_2(tmp_path):
    cfg = load_config(FIXTURES / "valid_config.yaml")
    cfg.input.path = FIXTURES / "ubiquitin.pdb"

    pipeline = Pipeline(cfg, output_root=tmp_path / "output")
    engine = _mock_engine()

    with patch("phosp.pipeline.shutil.which", return_value="/usr/bin/gmx"), \
         patch("phosp.pipeline.GROMACSEngine", return_value=engine), \
         patch("phosp.stages.stage1_modify.protonate_structure",
               side_effect=lambda p, o, ph: shutil.copy(p, o) or o), \
         patch("phosp.forcefields.charmm36m.CHARMM36mFF.patch_topology",
               side_effect=lambda top, sites: top):
        pipeline.execute(only_stages="1,2")

    assert (tmp_path / "output" / "stage1" / "modified.pdb").exists()
    assert (tmp_path / "output" / "stage2" / "topol.top").exists()
    assert pipeline.checkpoint.is_complete("stage1")
    assert pipeline.checkpoint.is_complete("stage2")


def test_pipeline_resumes_from_stage2(tmp_path):
    cfg = load_config(FIXTURES / "valid_config.yaml")
    cfg.input.path = FIXTURES / "ubiquitin.pdb"
    pipeline = Pipeline(cfg, output_root=tmp_path / "output")
    pipeline.checkpoint.mark_complete("stage1", {"modified_pdb": "fake.pdb"})

    stage1_dir = tmp_path / "output" / "stage1"
    stage1_dir.mkdir(parents=True)
    shutil.copy(FIXTURES / "ubiquitin.pdb", stage1_dir / "modified.pdb")
    with (stage1_dir / "modification_manifest.json").open("w") as f:
        json.dump([], f)

    engine = _mock_engine()
    with patch("phosp.pipeline.shutil.which", return_value="/usr/bin/gmx"), \
         patch("phosp.pipeline.GROMACSEngine", return_value=engine), \
         patch("phosp.forcefields.charmm36m.CHARMM36mFF.patch_topology",
               side_effect=lambda top, sites: top):
        pipeline.execute(only_stages="1,2")

    engine.prepare_topology.assert_called_once()
