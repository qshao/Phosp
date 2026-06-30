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
    assert "25000000" in content  # 50 ns at 2 fs timestep


def test_render_production_mdp_uses_sim_config(tmp_path):
    p = Protocol.load("globular_protein", _sim_config(production_time_ns=200.0))
    mdp = p.render_mdp("production", tmp_path)
    content = mdp.read_text()
    # 200 ns * 1e6 fs / 2 fs = 100,000,000 steps
    assert "100000000" in content
