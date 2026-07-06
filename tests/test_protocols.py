from pathlib import Path
import pytest
from phosp.protocols.protocol import Protocol
from phosp.config import SimulationConfig


def _sim_config(**kw):
    return SimulationConfig(**kw)


def test_load_named_preset():
    p = Protocol.load("globular_protein", _sim_config())
    assert p is not None


def test_load_unknown_preset_raises():
    with pytest.raises(FileNotFoundError):
        Protocol.load("nonexistent_preset", _sim_config())


def test_render_minimization_mdp(tmp_path):
    p = Protocol.load("globular_protein", _sim_config())
    mdp = p.render_mdp("minimization", tmp_path)
    assert mdp.exists()
    content = mdp.read_text()
    assert "steep" in content
    assert "50000" in content


def test_render_nvt_mdp(tmp_path):
    p = Protocol.load("globular_protein", _sim_config())
    mdp = p.render_mdp("nvt", tmp_path)
    content = mdp.read_text()
    assert "V-rescale" in content
    nsteps_line = next(l for l in content.splitlines() if l.strip().startswith("nsteps"))
    assert nsteps_line.split("=")[1].strip() == "250000"  # 500 ps at 2 fs timestep


def test_render_production_mdp_uses_sim_config(tmp_path):
    p = Protocol.load("globular_protein", _sim_config(production_time_ns=200.0))
    mdp = p.render_mdp("production", tmp_path)
    content = mdp.read_text()
    # 200 ns * 1e6 fs / 2 fs = 100,000,000 steps
    assert "100000000" in content


def test_default_cutoff_is_1_2nm(tmp_path):
    p = Protocol.load("globular_protein", _sim_config())
    content = p.render_mdp("nvt", tmp_path).read_text()
    assert "rvdw                 = 1.2" in content
    assert "rcoulomb             = 1.2" in content


def test_custom_rvdw_nm_overrides_default(tmp_path):
    p = Protocol({"rvdw_nm": 1.0, "nvt": {}}, _sim_config())
    content = p.render_mdp("nvt", tmp_path).read_text()
    assert "rvdw                 = 1.0" in content
    assert "rcoulomb             = 1.0" in content  # defaults to rvdw_nm if unset


def test_custom_rcoulomb_nm_independent_of_rvdw(tmp_path):
    p = Protocol({"rvdw_nm": 1.0, "rcoulomb_nm": 1.2, "nvt": {}}, _sim_config())
    content = p.render_mdp("nvt", tmp_path).read_text()
    assert "rvdw                 = 1.0" in content
    assert "rcoulomb             = 1.2" in content


def test_load_100ns_eq_preset_has_1nm_cutoff_and_100ns_phases(tmp_path):
    p = Protocol.load("globular_protein_100ns_eq", _sim_config())
    nvt_content = p.render_mdp("nvt", tmp_path).read_text()
    npt_content = p.render_mdp("npt", tmp_path).read_text()
    nvt_nsteps = next(l for l in nvt_content.splitlines() if l.strip().startswith("nsteps"))
    npt_nsteps = next(l for l in npt_content.splitlines() if l.strip().startswith("nsteps"))
    assert nvt_nsteps.split("=")[1].strip() == "50000000"  # 100 ns at 2 fs
    assert npt_nsteps.split("=")[1].strip() == "50000000"  # 100 ns at 2 fs
    assert "rvdw                 = 1.0" in nvt_content
    assert "rcoulomb             = 1.0" in nvt_content
