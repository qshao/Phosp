from __future__ import annotations
import json
import logging
import subprocess
from pathlib import Path

from phosp.runners.base import SimulationRunner

logger = logging.getLogger(__name__)


class SlurmRunner(SimulationRunner):
    """Generate a SLURM job script and optionally submit it."""

    def __init__(self, hpc_config) -> None:
        self._hpc = hpc_config

    def run(self, phases, restraint_phases, topology, structure, mdp_dir,
            output_dir, work_dir, engine, gpu_id):
        hpc = self._hpc
        resources = {
            "ntasks": hpc.ntasks,
            "gpus": hpc.gpus,
            "walltime": hpc.walltime,
            "partition": hpc.partition,
        }
        for phase in phases:
            (output_dir / phase).mkdir(parents=True, exist_ok=True)

        script = engine.generate_hpc_script(
            scheduler="slurm",
            resources=resources,
            phases=phases,
            output_dir=output_dir,
            work_dir=work_dir,
        )

        job_id: str | None = None
        if hpc.auto_submit:
            result = subprocess.run(
                ["sbatch", str(script)], capture_output=True, text=True, check=True
            )
            # sbatch prints: "Submitted batch job 12345"
            job_id = result.stdout.strip().split()[-1] if result.stdout.strip() else None
            logger.info("Submitted SLURM job %s: %s", job_id, script)
        else:
            logger.info("SLURM script written (not submitted): %s", script)

        sentinel = output_dir / "pending_job.json"
        sentinel.write_text(json.dumps({
            "scheduler": "slurm",
            "script": str(script),
            "job_id": job_id,
            "auto_submitted": hpc.auto_submit,
        }))

        return {"hpc_script": script}
