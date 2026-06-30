from __future__ import annotations
from phosp.runners.base import SimulationRunner


def get_runner(sim_config) -> SimulationRunner:
    runner = sim_config.runner
    if runner == "local":
        from phosp.runners.local import LocalRunner
        return LocalRunner()
    if runner == "slurm":
        from phosp.runners.slurm import SlurmRunner
        return SlurmRunner(sim_config.hpc)
    if runner == "pbs":
        from phosp.runners.pbs import PBSRunner
        return PBSRunner(sim_config.hpc)
    raise ValueError(f"Unknown runner: {runner!r}")
