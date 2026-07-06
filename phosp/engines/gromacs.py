from __future__ import annotations
import logging
import subprocess
from pathlib import Path

from phosp.engines.base import MDEngine, SimulationResult
from phosp.exceptions import SimulationError

logger = logging.getLogger(__name__)


def _run_gmx(
    cmd: list[str], cwd: Path, input_text: str = "", timeout: int | None = None
) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True,
            input=input_text, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        minutes = timeout // 60 if timeout else "?"
        raise SimulationError(
            f"GROMACS timed out after {minutes} minutes: {cmd[0]} {cmd[1]}"
        )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout)[-2000:]
        raise SimulationError(f"GROMACS command failed: {' '.join(cmd)}\n{tail}")
    return result


class GROMACSEngine(MDEngine):
    def __init__(self, binary: str = "gmx", timeout_minutes: int | None = None) -> None:
        self._binary = binary
        self._timeout = timeout_minutes * 60 if timeout_minutes is not None else None

    def prepare_topology(
        self,
        pdb: Path,
        forcefield,
        output_dir: Path | None = None,
        water_model: str = "tip3p",
        ff_flag_override: str | None = None,
    ) -> Path:
        if output_dir is None:
            output_dir = pdb.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        # ff_flag_override points pdb2gmx at a per-run extended force-field
        # directory (see CHARMM36mFF.build_ncaa_forcefield) instead of the
        # force field's own default flag — cwd=output_dir is what makes
        # GROMACS's cwd-first "-ff <name>" search find it there.
        ff_flag = ff_flag_override if ff_flag_override is not None else forcefield.pdb2gmx_flag()
        _run_gmx(
            [self._binary, "pdb2gmx",
             "-f", str(pdb.resolve()),
             "-o", str(output_dir / "processed.gro"),
             "-p", str(output_dir / "topol.top"),
             "-ff", ff_flag,
             "-water", water_model,
             "-ignh"],
            cwd=output_dir,
            timeout=self._timeout,
        )
        return output_dir / "topol.top"

    # TIP3P uses spc216.gro since GROMACS dropped tip3p.gro in 2024+
    _WATER_BOX = {"tip3p": "spc216", "spc": "spc216", "spce": "spc216",
                  "tip4p": "tip4p", "tip5p": "tip5p"}

    def solvate(self, gro: Path, topology: Path, box_type: str, water_model: str) -> tuple[Path, Path]:
        out_dir = gro.parent
        box_gro = out_dir / "newbox.gro"
        solvated_gro = out_dir / "solvated.gro"
        water_box = self._WATER_BOX.get(water_model.lower(), water_model)
        _run_gmx(
            [self._binary, "editconf", "-f", str(gro), "-o", str(box_gro),
             "-c", "-d", "1.2", "-bt", box_type],
            cwd=out_dir,
            timeout=self._timeout,
        )
        _run_gmx(
            [self._binary, "solvate", "-cp", str(box_gro), "-cs", f"{water_box}.gro",
             "-o", str(solvated_gro), "-p", str(topology)],
            cwd=out_dir,
            timeout=self._timeout,
        )
        return solvated_gro, topology

    def add_ions(self, gro: Path, topology: Path, concentration_mM: float, neutralize: bool) -> tuple[Path, Path]:
        out_dir = gro.parent
        ions_tpr = out_dir / "ions.tpr"
        ions_gro = out_dir / "ions.gro"
        genion_mdp = out_dir / "genion.mdp"
        genion_mdp.write_text("integrator=steep\nnsteps=0\n")
        _run_gmx(
            [self._binary, "grompp", "-f", str(genion_mdp), "-c", str(gro),
             "-p", str(topology), "-o", str(ions_tpr), "-maxwarn", "2"],
            cwd=out_dir,
            timeout=self._timeout,
        )
        conc = concentration_mM / 1000.0
        neutral_flag = ["-neutral"] if neutralize else []
        _run_gmx(
            [self._binary, "genion", "-s", str(ions_tpr), "-o", str(ions_gro),
             "-p", str(topology), "-pname", "NA", "-nname", "CL",
             "-conc", str(conc)] + neutral_flag,
            cwd=out_dir,
            input_text="SOL\n",
            timeout=self._timeout,
        )
        return ions_gro, topology

    def generate_mdp(self, phase: str, protocol, output_dir: Path) -> Path:
        return protocol.render_mdp(phase, output_dir)

    def run_phase(
        self,
        phase: str,
        mdp: Path,
        topology: Path,
        structure: Path,
        output_dir: Path,
        restraint_gro: Path | None = None,
        gpu_id: int | None = None,
    ) -> SimulationResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        tpr = output_dir / f"{phase}.tpr"
        log = output_dir / f"{phase}.log"

        grompp_cmd = [
            self._binary, "grompp",
            "-f", str(mdp),
            "-c", str(structure),
            "-p", str(topology),
            "-o", str(tpr),
            "-maxwarn", "2",
        ]
        if restraint_gro:
            grompp_cmd += ["-r", str(restraint_gro)]
        _run_gmx(grompp_cmd, cwd=output_dir, timeout=self._timeout)

        mdrun_cmd = [
            self._binary, "mdrun", "-v",
            "-deffnm", str(output_dir / phase),
            "-ntmpi", "1",
        ]
        if gpu_id is not None:
            mdrun_cmd += ["-gpu_id", str(gpu_id), "-nb", "gpu", "-pme", "gpu", "-bonded", "gpu"]
            if phase != "minimization":
                # GPU-resident update (-update gpu) requires a dynamical integrator
                # (md/md-vv/sd); minimization uses steep and GROMACS fatal-errors if
                # -update gpu is passed to it. -nb/-pme/-bonded gpu are pure
                # per-step force-calculation offloads, independent of the
                # integrator, so they're valid (and desirable) for minimization too.
                mdrun_cmd += ["-update", "gpu"]
        _run_gmx(mdrun_cmd, cwd=output_dir, timeout=self._timeout)

        return SimulationResult(
            phase=phase,
            output_dir=output_dir,
            success=True,
            log_path=log,
        )

    def generate_hpc_script(
        self,
        scheduler: str,
        resources: dict,
        phases: list[str],
        output_dir: Path,
        work_dir: Path | None = None,
        gpu_id: int | None = None,
    ) -> Path:
        from jinja2 import Environment, FileSystemLoader
        templates_dir = Path(__file__).parent.parent / "templates"
        env = Environment(loader=FileSystemLoader(str(templates_dir)))
        template = env.get_template(f"{scheduler}_job.sh.j2")
        rendered = template.render(
            resources=resources,
            phases=phases,
            output_dir=str(work_dir or output_dir),
            binary=self._binary,
            gpu_id=gpu_id,
        )
        script = output_dir / f"run_{scheduler}.sh"
        script.write_text(rendered)
        script.chmod(0o755)
        return script
