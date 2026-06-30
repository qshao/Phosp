from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from phosp.engines.gromacs import GROMACSEngine, _run_gmx
from phosp.exceptions import SimulationError


def test_run_gmx_raises_on_nonzero(tmp_path):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="Fatal error", stdout="")
        with pytest.raises(SimulationError, match="Fatal error"):
            _run_gmx(["gmx", "help"], cwd=tmp_path)


def test_run_gmx_returns_completed_process(tmp_path):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="ok")
        result = _run_gmx(["gmx", "help"], cwd=tmp_path)
        assert result.returncode == 0


def test_generate_mdp_delegates_to_protocol(tmp_path):
    engine = GROMACSEngine()
    protocol_data = {"minimization": {"integrator": "steep", "nsteps": 50000, "emtol": 1000.0, "emstep": 0.01}}
    from phosp.config import SimulationConfig
    from phosp.protocols.protocol import Protocol
    proto = Protocol(protocol_data, SimulationConfig())
    mdp = engine.generate_mdp("minimization", proto, tmp_path)
    assert mdp.exists()
    assert "steep" in mdp.read_text()
