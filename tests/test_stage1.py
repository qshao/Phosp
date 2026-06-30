import json
from pathlib import Path
from unittest.mock import patch
import pytest
from phosp.config import load_config
from phosp.stages.stage1_modify import Stage1Modify

FIXTURES = Path(__file__).parent / "fixtures"


def _make_stage(tmp_path, cfg_overrides=None):
    cfg = load_config(FIXTURES / "valid_config.yaml")
    # Point input path to fixture
    cfg.input.path = FIXTURES / "ubiquitin.pdb"
    return Stage1Modify(cfg, engine=None, forcefield=None, output_root=tmp_path / "stage1")


def test_stage1_creates_modified_pdb(tmp_path):
    stage = _make_stage(tmp_path)
    # Mock protonate_structure to avoid requiring pdb2pqr in CI
    with patch("phosp.stages.stage1_modify.protonate_structure", side_effect=lambda p, o, ph, pdb2pqr_binary="pdb2pqr", timeout=None: (o.parent / "input.pdb").rename(o) or o):
        result = stage.run()
    assert (tmp_path / "stage1" / "modified.pdb").exists()
    assert result.stage == "stage1"


def test_stage1_writes_manifest(tmp_path):
    stage = _make_stage(tmp_path)
    with patch("phosp.stages.stage1_modify.protonate_structure", side_effect=lambda p, o, ph, pdb2pqr_binary="pdb2pqr", timeout=None: (o.parent / "input.pdb").rename(o) or o):
        stage.run()
    manifest = json.loads((tmp_path / "stage1" / "modification_manifest.json").read_text())
    assert isinstance(manifest, list)
    assert manifest[0]["phospho_type"] == "pThr"


def test_stage1_validate_inputs_missing_pdb(tmp_path):
    from phosp.exceptions import StageInputError
    cfg = load_config(FIXTURES / "valid_config.yaml")
    cfg.input.path = Path("nonexistent.pdb")
    stage = Stage1Modify(cfg, engine=None, forcefield=None, output_root=tmp_path)
    with pytest.raises(StageInputError, match="not found"):
        stage.validate_inputs()


def test_stage1_validate_inputs_bad_site(tmp_path):
    from phosp.exceptions import StageInputError
    cfg = load_config(FIXTURES / "valid_config.yaml")
    cfg.input.path = FIXTURES / "ubiquitin.pdb"
    # Mutate site to use a chain that doesn't exist
    cfg.modification.sites[0].chain = "Z"
    stage = Stage1Modify(cfg, engine=None, forcefield=None, output_root=tmp_path)
    with pytest.raises(StageInputError, match="chain Z"):
        stage.validate_inputs()


def test_stage1_validate_inputs_bad_resid(tmp_path):
    from phosp.exceptions import StageInputError
    cfg = load_config(FIXTURES / "valid_config.yaml")
    cfg.input.path = FIXTURES / "ubiquitin.pdb"
    # Mutate site to use a resid that doesn't exist in chain A
    cfg.modification.sites[0].resid = 9999
    stage = Stage1Modify(cfg, engine=None, forcefield=None, output_root=tmp_path)
    with pytest.raises(StageInputError, match="resid 9999"):
        stage.validate_inputs()


def test_stage1_reference_mode_skips_modification(tmp_path):
    """In reference mode, no phospho modification is applied, manifest is empty."""
    stage = Stage1Modify(
        load_config(FIXTURES / "valid_config.yaml"),
        engine=None,
        forcefield=None,
        output_root=tmp_path / "stage1",
        reference_mode=True,
    )
    # Point input to fixture
    stage.config.input.path = FIXTURES / "ubiquitin.pdb"
    with patch("phosp.stages.stage1_modify.protonate_structure",
               side_effect=lambda p, o, ph, pdb2pqr_binary="pdb2pqr", timeout=None: (o.parent / "input.pdb").rename(o) or o):
        result = stage.run()
    manifest = json.loads((tmp_path / "stage1" / "modification_manifest.json").read_text())
    assert manifest == [], "reference mode must produce empty manifest"
    assert (tmp_path / "stage1" / "modified.pdb").exists()


def test_stage1_reference_mode_skips_site_validation(tmp_path):
    """In reference mode, site validation is skipped even for bad sites."""
    cfg = load_config(FIXTURES / "valid_config.yaml")
    cfg.input.path = FIXTURES / "ubiquitin.pdb"
    cfg.modification.sites[0].chain = "Z"  # bad chain — would normally raise
    stage = Stage1Modify(cfg, engine=None, forcefield=None, output_root=tmp_path, reference_mode=True)
    # Should NOT raise StageInputError
    stage.validate_inputs()


def test_stage1_passes_gromacs_timeout_to_protonate_structure(tmp_path):
    """gromacs.timeout_minutes bounds the pdb2pqr subprocess so it can't hang forever."""
    stage = _make_stage(tmp_path)
    stage.config.gromacs.timeout_minutes = 5
    with patch("phosp.stages.stage1_modify.protonate_structure") as mock_protonate:
        mock_protonate.side_effect = lambda p, o, ph, pdb2pqr_binary="pdb2pqr", timeout=None: (o.parent / "input.pdb").rename(o) or o
        stage.run()
    assert mock_protonate.call_args.kwargs["timeout"] == 300
