from __future__ import annotations
import json
import logging
from pathlib import Path

from phosp.config import ModificationConfig
from phosp.exceptions import StageInputError
from phosp.protocols.protocol import Protocol
from phosp.stages.base import Stage, StageResult

logger = logging.getLogger(__name__)


class Stage2Prepare(Stage):
    def validate_inputs(self) -> None:
        modified_pdb = self.output_root.parent / "stage1" / "modified.pdb"
        if not modified_pdb.exists():
            raise StageInputError(f"modified.pdb not found: {modified_pdb}. Run stage1 first.")

    def run(self) -> StageResult:
        out = self.output_root
        cfg = self.config
        sim = cfg.simulation

        stage1_dir = out.parent / "stage1"
        modified_pdb = stage1_dir / "modified.pdb"
        manifest = json.loads((stage1_dir / "modification_manifest.json").read_text())
        sites = cfg.modification.sites

        # 1. Build topology
        topology = self.engine.prepare_topology(
            modified_pdb,
            self.forcefield,
            output_dir=out,
            water_model=sim.water_model,
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
            "phospho_sites": manifest,
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
