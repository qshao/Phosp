from pathlib import Path
import pytest
from pydantic import ValidationError
from phosp.config import PhospConfig, ModificationSite, load_config, SimulationConfig, HPCConfig, AnalysisConfig

FIXTURES = Path(__file__).parent / "fixtures"

def test_analysis_config_rmsd_defaults_to_ca_only():
    """Project default is CA-only RMSD (see rmsd.py) — the config default must match,
    otherwise every config that doesn't set analysis.rmsd.selection explicitly
    silently gets full-backbone RMSD instead of the documented default."""
    assert AnalysisConfig().rmsd == {"selection": "name CA", "reference": "first_frame"}

def test_load_valid_config():
    cfg = load_config(FIXTURES / "valid_config.yaml")
    assert cfg.forcefield == "charmm36m"
    assert cfg.input.source == "pdb"
    assert cfg.input.ph == 7.4

def test_modificationsite_resname_mismatch():
    with pytest.raises(Exception, match="requires resname"):
        ModificationSite(chain="A", resid=42, resname="SER", mod_type="pThr")


def test_modificationsite_unknown_mod_type():
    with pytest.raises(Exception, match="Unknown mod_type"):
        ModificationSite(chain="A", resid=42, resname="SER", mod_type="pHis")

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
    assert cfg.simulation.runner == "local"


def test_production_time_must_be_positive():
    with pytest.raises(ValidationError, match="production_time_ns"):
        SimulationConfig(production_time_ns=0.0)


def test_production_time_negative_rejected():
    with pytest.raises(ValidationError, match="production_time_ns"):
        SimulationConfig(production_time_ns=-5.0)


def test_output_freq_must_be_positive():
    with pytest.raises(ValidationError, match="output_freq_ps"):
        SimulationConfig(output_freq_ps=0.0)


def test_output_freq_cannot_exceed_production():
    with pytest.raises(ValidationError, match="output_freq_ps"):
        SimulationConfig(production_time_ns=1.0, output_freq_ps=2000.0)


def test_valid_output_freq_equal_to_production():
    cfg = SimulationConfig(production_time_ns=100.0, output_freq_ps=100000.0)
    assert cfg.output_freq_ps == 100000.0


def test_salt_concentration_non_negative():
    with pytest.raises(ValidationError, match="salt_concentration_mM"):
        SimulationConfig(salt_concentration_mM=-1.0)


def test_salt_concentration_zero_allowed():
    cfg = SimulationConfig(salt_concentration_mM=0.0)
    assert cfg.salt_concentration_mM == 0.0


def test_hpc_ntasks_must_be_at_least_1():
    with pytest.raises(ValidationError, match="ntasks"):
        HPCConfig(ntasks=0)


def test_hpc_gpus_non_negative():
    with pytest.raises(ValidationError, match="gpus"):
        HPCConfig(gpus=-1)
