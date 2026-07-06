from __future__ import annotations
import json
import logging
from pathlib import Path

from phosp.config import ModificationConfig
from phosp.exceptions import PhospError, StageInputError
from phosp.forcefields.discovery import discover_top_dir
from phosp.protocols.protocol import Protocol
from phosp.stages.base import Stage, StageResult

logger = logging.getLogger(__name__)


class Stage2Prepare(Stage):
    def validate_inputs(self) -> None:
        stage1_dir = self.output_root.parent / "stage1"
        for fname in ("modified.pdb", "modification_manifest.json"):
            if not (stage1_dir / fname).exists():
                raise StageInputError(f"{fname} not found in {stage1_dir}. Run stage1 first.")

    def run(self) -> StageResult:
        out = self.output_root
        out.mkdir(parents=True, exist_ok=True)
        cfg = self.config
        sim = cfg.simulation

        stage1_dir = out.parent / "stage1"
        modified_pdb = stage1_dir / "modified.pdb"
        manifest = json.loads((stage1_dir / "modification_manifest.json").read_text())
        sites = cfg.modification.sites
        ncaa_sites = cfg.modification.ncaa_sites

        # 1. Build topology — if there are ncAA sites, first build a per-run
        # extended force-field directory (residue bundles merged in) and
        # point pdb2gmx at it instead of the force field's default flag.
        ff_flag_override = None
        if ncaa_sites:
            ff_dirname = f"{self.forcefield.pdb2gmx_flag()}.ff"
            top_dir = discover_top_dir(cfg.gromacs.binary, ff_dirname)
            if top_dir is None or not (top_dir / ff_dirname).exists():
                raise PhospError(
                    f"ncAA sites are configured but the base force field "
                    f"directory {ff_dirname} could not be located (checked "
                    f"GMXLIB and the gmx Data prefix)."
                )
            ff_flag_override = self.forcefield.build_ncaa_forcefield(
                bundle_dirs=[site.bundle_dir for site in ncaa_sites],
                base_ff_dir=top_dir / ff_dirname,
                output_dir=out,
            )

        topology = self.engine.prepare_topology(
            modified_pdb,
            self.forcefield,
            output_dir=out,
            water_model=sim.water_model,
            ff_flag_override=ff_flag_override,
        )
        topology = self.forcefield.patch_topology(topology, sites)

        # 2. Solvate
        processed_gro = topology.parent / "processed.gro"
        solvated_gro, topology = self.engine.solvate(
            processed_gro, topology,
            box_type=sim.box_type,
            water_model=sim.water_model,
        )

        # 3. Add ions
        ions_gro, topology = self.engine.add_ions(
            solvated_gro, topology,
            concentration_mM=sim.salt_concentration_mM,
            neutralize=True,
        )

        # 4. Generate MDP files
        protocol = Protocol.load(cfg.protocol, sim)
        mdp_files = {}
        for phase in ["minimization", "nvt", "npt", "production"]:
            mdp_files[phase] = self.engine.generate_mdp(phase, protocol, out)

        # 5. Write prep report
        report = {
            "forcefield": cfg.forcefield,
            "protocol": cfg.protocol,
            "water_model": sim.water_model,
            "box_type": sim.box_type,
            "salt_mM": sim.salt_concentration_mM,
            "modification_sites": manifest,
        }
        report_path = out / "prep_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        return StageResult(
            stage="stage2",
            output_dir=out,
            artifacts={
                "topology": topology,
                "structure": ions_gro,
                "prep_report": report_path,
                **{f"mdp_{p}": mdp_files[p] for p in mdp_files},
            },
        )
