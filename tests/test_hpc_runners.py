from pathlib import Path
from unittest.mock import MagicMock
from phosp.runners.pbs import PBSRunner
from phosp.runners.slurm import SlurmRunner
from phosp.config import HPCConfig


def _hpc_config(**overrides):
    return HPCConfig(**{"auto_submit": False, **overrides})


def test_pbs_runner_forwards_gpu_id_to_generate_hpc_script(tmp_path):
    """PBSRunner must forward the configured gpu_id, not let the template
    hardcode GPU 0 regardless of what the user set."""
    engine = MagicMock()
    engine.generate_hpc_script.return_value = tmp_path / "run_pbs.sh"
    runner = PBSRunner(_hpc_config())
    runner.run(
        phases=["minimization"], restraint_phases=set(),
        topology=tmp_path / "topol.top", structure=tmp_path / "ions.gro",
        mdp_dir=tmp_path, output_dir=tmp_path, work_dir=tmp_path,
        engine=engine, gpu_id=1,
    )
    assert engine.generate_hpc_script.call_args.kwargs["gpu_id"] == 1


def test_slurm_runner_forwards_gpu_id_to_generate_hpc_script(tmp_path):
    engine = MagicMock()
    engine.generate_hpc_script.return_value = tmp_path / "run_slurm.sh"
    runner = SlurmRunner(_hpc_config())
    runner.run(
        phases=["minimization"], restraint_phases=set(),
        topology=tmp_path / "topol.top", structure=tmp_path / "ions.gro",
        mdp_dir=tmp_path, output_dir=tmp_path, work_dir=tmp_path,
        engine=engine, gpu_id=2,
    )
    assert engine.generate_hpc_script.call_args.kwargs["gpu_id"] == 2
