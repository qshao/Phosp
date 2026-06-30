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


def test_run_phase_returns_simulation_result(tmp_path):
    engine = GROMACSEngine()
    phase_dir = tmp_path / "nvt"
    phase_dir.mkdir()
    fake_tpr = phase_dir / "nvt.tpr"
    fake_log = phase_dir / "nvt.log"
    fake_log.write_text("Finished mdrun")

    with patch("phosp.engines.gromacs._run_gmx") as mock_gmx:
        mock_gmx.return_value = MagicMock(returncode=0)
        fake_tpr.write_text("")
        result = engine.run_phase(
            phase="nvt",
            mdp=tmp_path / "nvt.mdp",
            topology=tmp_path / "topol.top",
            structure=tmp_path / "ions.gro",
            output_dir=phase_dir,
            restraint_gro=tmp_path / "ions.gro",
        )
    assert result.phase == "nvt"
    assert result.success is True


def test_generate_slurm_script(tmp_path):
    engine = GROMACSEngine()
    work_dir = tmp_path / "stage3"
    script = engine.generate_hpc_script(
        scheduler="slurm",
        resources={"ntasks": 8, "gpus": 1, "walltime": "24:00:00", "partition": "gpu"},
        phases=["minimization", "nvt", "npt", "production"],
        output_dir=tmp_path,
        work_dir=work_dir,
    )
    assert script.exists()
    content = script.read_text()
    assert "#SBATCH" in content
    assert "gmx mdrun" in content
    # MDP files live in stage2/, not stage3/
    assert "../stage2/minimization.mdp" in content
    assert str(work_dir) in content
