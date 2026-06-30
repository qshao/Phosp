from pathlib import Path
import pytest
from phosp.config import PhospConfig, PhosphoSite, load_config

FIXTURES = Path(__file__).parent / "fixtures"

def test_load_valid_config():
    cfg = load_config(FIXTURES / "valid_config.yaml")
    assert cfg.forcefield == "charmm36m"
    assert cfg.input.source == "pdb"
    assert cfg.input.ph == 7.4

def test_phosphosite_resname_mismatch():
    with pytest.raises(Exception, match="must use"):
        PhosphoSite(chain="A", resid=42, resname="SER", phospho_type="pThr")

def test_missing_path_for_pdb_source():
    with pytest.raises(Exception):
        PhospConfig.model_validate({
            "input": {"source": "pdb"},
            "modification": {"sites": []},
        })

def test_default_simulation_values():
    cfg = load_config(FIXTURES / "valid_config.yaml")
    assert cfg.simulation.production_time_ns == 100.0
    assert cfg.simulation.salt_concentration_mM == 150.0
    assert cfg.simulation.hpc.enabled is False
