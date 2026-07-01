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


def test_run_phase_passes_gpu_id_to_mdrun(tmp_path):
    engine = GROMACSEngine()
    phase_dir = tmp_path / "nvt"
    phase_dir.mkdir()
    (phase_dir / "nvt.log").write_text("done")

    with patch("phosp.engines.gromacs._run_gmx") as mock_gmx:
        mock_gmx.return_value = MagicMock(returncode=0)
        engine.run_phase(
            phase="nvt",
            mdp=tmp_path / "nvt.mdp",
            topology=tmp_path / "topol.top",
            structure=tmp_path / "ions.gro",
            output_dir=phase_dir,
            gpu_id=0,
        )
    mdrun_call = mock_gmx.call_args_list[1]  # second call is mdrun
    assert "-gpu_id" in mdrun_call.args[0]
    assert "0" in mdrun_call.args[0]


def test_run_phase_omits_gpu_flag_when_gpu_id_is_none(tmp_path):
    engine = GROMACSEngine()
    phase_dir = tmp_path / "nvt"
    phase_dir.mkdir()
    (phase_dir / "nvt.log").write_text("done")

    with patch("phosp.engines.gromacs._run_gmx") as mock_gmx:
        mock_gmx.return_value = MagicMock(returncode=0)
        engine.run_phase(
            phase="nvt",
            mdp=tmp_path / "nvt.mdp",
            topology=tmp_path / "topol.top",
            structure=tmp_path / "ions.gro",
            output_dir=phase_dir,
            gpu_id=None,
        )
    mdrun_call = mock_gmx.call_args_list[1]
    assert "-gpu_id" not in mdrun_call.args[0]


def test_run_phase_offloads_pme_bonded_update_to_gpu_when_gpu_configured(tmp_path):
    """A configured GPU (e.g. A100/H100/H200) should also take PME, bonded,
    and update work, not just nonbonded — otherwise a fast GPU sits mostly idle."""
    engine = GROMACSEngine()
    phase_dir = tmp_path / "nvt"
    phase_dir.mkdir()
    (phase_dir / "nvt.log").write_text("done")

    with patch("phosp.engines.gromacs._run_gmx") as mock_gmx:
        mock_gmx.return_value = MagicMock(returncode=0)
        engine.run_phase(
            phase="nvt",
            mdp=tmp_path / "nvt.mdp",
            topology=tmp_path / "topol.top",
            structure=tmp_path / "ions.gro",
            output_dir=phase_dir,
            gpu_id=0,
        )
    mdrun_args = mock_gmx.call_args_list[1].args[0]
    assert mdrun_args[mdrun_args.index("-nb") + 1] == "gpu"
    assert mdrun_args[mdrun_args.index("-pme") + 1] == "gpu"
    assert mdrun_args[mdrun_args.index("-bonded") + 1] == "gpu"
    assert mdrun_args[mdrun_args.index("-update") + 1] == "gpu"


def test_run_phase_omits_gpu_offload_flags_when_gpu_id_is_none(tmp_path):
    engine = GROMACSEngine()
    phase_dir = tmp_path / "nvt"
    phase_dir.mkdir()
    (phase_dir / "nvt.log").write_text("done")

    with patch("phosp.engines.gromacs._run_gmx") as mock_gmx:
        mock_gmx.return_value = MagicMock(returncode=0)
        engine.run_phase(
            phase="nvt",
            mdp=tmp_path / "nvt.mdp",
            topology=tmp_path / "topol.top",
            structure=tmp_path / "ions.gro",
            output_dir=phase_dir,
            gpu_id=None,
        )
    mdrun_args = mock_gmx.call_args_list[1].args[0]
    for flag in ("-nb", "-pme", "-bonded", "-update"):
        assert flag not in mdrun_args


def _slurm_resources(**overrides):
    base = {
        "ntasks": 8, "gpus": 1, "walltime": "24:00:00",
        "partition": "gpu", "gromacs_module": None, "extra_directives": [],
    }
    return {**base, **overrides}


def test_generate_slurm_script(tmp_path):
    engine = GROMACSEngine()
    work_dir = tmp_path / "stage3"
    script = engine.generate_hpc_script(
        scheduler="slurm",
        resources=_slurm_resources(),
        phases=["minimization", "nvt", "npt", "production"],
        output_dir=tmp_path,
        work_dir=work_dir,
    )
    assert script.exists()
    content = script.read_text()
    assert "#SBATCH" in content
    assert "--cpus-per-task=8" in content   # not --ntasks
    assert "--ntasks=1" in content
    assert "gmx mdrun" in content
    assert "../stage2/minimization.mdp" in content
    assert str(work_dir) in content


def test_slurm_script_log_paths_are_absolute(tmp_path):
    engine = GROMACSEngine()
    work_dir = tmp_path / "stage3"
    script = engine.generate_hpc_script(
        scheduler="slurm",
        resources=_slurm_resources(),
        phases=["minimization"],
        output_dir=tmp_path,
        work_dir=work_dir,
    )
    content = script.read_text()
    # Log paths must be absolute so they land in the job dir, not wherever sbatch is run
    assert f"--output={work_dir}/slurm_" in content
    assert f"--error={work_dir}/slurm_" in content


def test_slurm_script_module_load_when_configured(tmp_path):
    engine = GROMACSEngine()
    script = engine.generate_hpc_script(
        scheduler="slurm",
        resources=_slurm_resources(gromacs_module="gromacs/2026.0-cuda"),
        phases=["minimization"],
        output_dir=tmp_path,
    )
    assert "module load gromacs/2026.0-cuda" in script.read_text()


def test_slurm_script_no_module_load_when_none(tmp_path):
    engine = GROMACSEngine()
    script = engine.generate_hpc_script(
        scheduler="slurm",
        resources=_slurm_resources(gromacs_module=None),
        phases=["minimization"],
        output_dir=tmp_path,
    )
    assert "module load" not in script.read_text()


def test_slurm_script_gpu_id_propagated(tmp_path):
    engine = GROMACSEngine()
    script = engine.generate_hpc_script(
        scheduler="slurm",
        resources=_slurm_resources(),
        phases=["minimization"],
        output_dir=tmp_path,
        gpu_id=1,
    )
    assert "-gpu_id 1" in script.read_text()


def test_slurm_script_no_gpu_id_when_none(tmp_path):
    engine = GROMACSEngine()
    script = engine.generate_hpc_script(
        scheduler="slurm",
        resources=_slurm_resources(),
        phases=["minimization"],
        output_dir=tmp_path,
        gpu_id=None,
    )
    assert "-gpu_id" not in script.read_text()


def test_generate_slurm_script_custom_binary(tmp_path):
    engine = GROMACSEngine(binary="gmx_mpi")
    work_dir = tmp_path / "stage3"
    script = engine.generate_hpc_script(
        scheduler="slurm",
        resources=_slurm_resources(),
        phases=["minimization", "nvt", "npt", "production"],
        output_dir=tmp_path,
        work_dir=work_dir,
    )
    content = script.read_text()
    assert "gmx_mpi grompp" in content
    # MPI build: mdrun is launched via srun, not with -ntmpi
    assert "srun" in content
    assert "gmx_mpi mdrun" in content
    assert "-ntmpi" not in content


def test_slurm_script_non_mpi_uses_ntmpi(tmp_path):
    engine = GROMACSEngine(binary="gmx")
    script = engine.generate_hpc_script(
        scheduler="slurm",
        resources=_slurm_resources(),
        phases=["minimization"],
        output_dir=tmp_path,
    )
    content = script.read_text()
    assert "-ntmpi 1" in content
    assert "srun" not in content


def test_slurm_script_extra_directives(tmp_path):
    engine = GROMACSEngine()
    script = engine.generate_hpc_script(
        scheduler="slurm",
        resources=_slurm_resources(extra_directives=[
            "--account=myproject",
            "--qos=high",
            "--constraint=a100",
        ]),
        phases=["minimization"],
        output_dir=tmp_path,
    )
    content = script.read_text()
    assert "#SBATCH --account=myproject" in content
    assert "#SBATCH --qos=high" in content
    assert "#SBATCH --constraint=a100" in content


def test_slurm_script_no_extra_directives_by_default(tmp_path):
    engine = GROMACSEngine()
    script = engine.generate_hpc_script(
        scheduler="slurm",
        resources=_slurm_resources(),
        phases=["minimization"],
        output_dir=tmp_path,
    )
    # Only standard #SBATCH lines — no unexpected extras
    sbatch_lines = [l for l in script.read_text().splitlines() if l.startswith("#SBATCH")]
    standard_keys = {"--job-name", "--nodes", "--ntasks", "--cpus-per-task",
                     "--gres", "--time", "--partition", "--output", "--error"}
    for line in sbatch_lines:
        key = line.split()[1].split("=")[0]
        assert key in standard_keys, f"Unexpected #SBATCH directive: {line}"


def _pbs_resources(**overrides):
    base = {
        "ntasks": 8, "gpus": 1, "walltime": "24:00:00",
        "partition": "gpu", "gromacs_module": None, "extra_directives": [],
    }
    return {**base, **overrides}


def test_pbs_script_gpu_id_propagated(tmp_path):
    engine = GROMACSEngine()
    script = engine.generate_hpc_script(
        scheduler="pbs",
        resources=_pbs_resources(),
        phases=["minimization"],
        output_dir=tmp_path,
        gpu_id=1,
    )
    assert "-gpu_id 1" in script.read_text()


def test_pbs_script_no_gpu_id_when_none(tmp_path):
    engine = GROMACSEngine()
    script = engine.generate_hpc_script(
        scheduler="pbs",
        resources=_pbs_resources(),
        phases=["minimization"],
        output_dir=tmp_path,
        gpu_id=None,
    )
    assert "-gpu_id" not in script.read_text()


def test_pbs_script_gpu_offload_flags_when_gpu_id_set(tmp_path):
    engine = GROMACSEngine()
    script = engine.generate_hpc_script(
        scheduler="pbs",
        resources=_pbs_resources(),
        phases=["minimization"],
        output_dir=tmp_path,
        gpu_id=0,
    )
    content = script.read_text()
    assert "-nb gpu" in content
    assert "-pme gpu" in content
    assert "-bonded gpu" in content
    assert "-update gpu" in content


def test_pbs_script_no_gpu_offload_flags_when_gpu_id_none(tmp_path):
    engine = GROMACSEngine()
    script = engine.generate_hpc_script(
        scheduler="pbs",
        resources=_pbs_resources(),
        phases=["minimization"],
        output_dir=tmp_path,
        gpu_id=None,
    )
    content = script.read_text()
    for flag in ("-nb gpu", "-pme gpu", "-bonded gpu", "-update gpu"):
        assert flag not in content


def test_slurm_script_gpu_offload_flags_when_gpu_id_set(tmp_path):
    engine = GROMACSEngine()
    script = engine.generate_hpc_script(
        scheduler="slurm",
        resources=_slurm_resources(),
        phases=["minimization"],
        output_dir=tmp_path,
        gpu_id=0,
    )
    content = script.read_text()
    assert "-nb gpu" in content
    assert "-pme gpu" in content
    assert "-bonded gpu" in content
    assert "-update gpu" in content


def test_slurm_script_no_gpu_offload_flags_when_gpu_id_none(tmp_path):
    engine = GROMACSEngine()
    script = engine.generate_hpc_script(
        scheduler="slurm",
        resources=_slurm_resources(),
        phases=["minimization"],
        output_dir=tmp_path,
        gpu_id=None,
    )
    content = script.read_text()
    for flag in ("-nb gpu", "-pme gpu", "-bonded gpu", "-update gpu"):
        assert flag not in content


def test_run_gmx_raises_simulation_error_on_timeout(tmp_path):
    """TimeoutExpired is caught and re-raised as SimulationError with 'timed out'."""
    import subprocess
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["gmx"], 60)):
        with pytest.raises(SimulationError, match="timed out after 1 minutes"):
            _run_gmx(["gmx", "mdrun"], cwd=tmp_path, timeout=60)


def test_engine_passes_timeout_to_run_gmx(tmp_path):
    """GROMACSEngine with timeout_minutes=2 passes 120s to _run_gmx."""
    engine = GROMACSEngine(binary="gmx", timeout_minutes=2)
    phase_dir = tmp_path / "min"
    phase_dir.mkdir()
    (phase_dir / "min.log").write_text("done")

    with patch("phosp.engines.gromacs._run_gmx") as mock_gmx:
        mock_gmx.return_value = MagicMock(returncode=0)
        engine.run_phase(
            phase="min",
            mdp=tmp_path / "min.mdp",
            topology=tmp_path / "topol.top",
            structure=tmp_path / "ions.gro",
            output_dir=phase_dir,
        )
    for call in mock_gmx.call_args_list:
        assert call.kwargs.get("timeout") == 120


def test_engine_no_timeout_by_default(tmp_path):
    """GROMACSEngine with no timeout_minutes passes timeout=None to _run_gmx."""
    engine = GROMACSEngine()
    phase_dir = tmp_path / "min"
    phase_dir.mkdir()
    (phase_dir / "min.log").write_text("done")

    with patch("phosp.engines.gromacs._run_gmx") as mock_gmx:
        mock_gmx.return_value = MagicMock(returncode=0)
        engine.run_phase(
            phase="min",
            mdp=tmp_path / "min.mdp",
            topology=tmp_path / "topol.top",
            structure=tmp_path / "ions.gro",
            output_dir=phase_dir,
        )
    for call in mock_gmx.call_args_list:
        assert call.kwargs.get("timeout") is None


def test_engine_uses_configured_binary(tmp_path):
    engine = GROMACSEngine(binary="/opt/gromacs/bin/gmx")
    phase_dir = tmp_path / "min"
    phase_dir.mkdir()
    (phase_dir / "min.log").write_text("done")

    with patch("phosp.engines.gromacs._run_gmx") as mock_gmx:
        mock_gmx.return_value = MagicMock(returncode=0)
        engine.run_phase(
            phase="min",
            mdp=tmp_path / "min.mdp",
            topology=tmp_path / "topol.top",
            structure=tmp_path / "ions.gro",
            output_dir=phase_dir,
        )
    for call in mock_gmx.call_args_list:
        assert call.args[0][0] == "/opt/gromacs/bin/gmx"
