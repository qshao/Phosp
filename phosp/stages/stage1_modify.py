from __future__ import annotations
import json
import logging
import shutil
from pathlib import Path

from Bio.PDB import PDBIO

from phosp.exceptions import StageInputError
from phosp.modification.base import get_modifier
from phosp.stages.base import Stage, StageResult
from phosp.utils.structure import clean_structure, fetch_structure, protonate_structure

logger = logging.getLogger(__name__)


class Stage1Modify(Stage):
    def validate_inputs(self) -> None:
        src = self.config.input
        if src.source == "pdb" and not src.path.exists():
            raise StageInputError(f"Input PDB not found: {src.path}")

    def run(self) -> StageResult:
        out = self.output_root
        cfg = self.config

        # 1. Acquire structure
        raw = fetch_structure(
            source=cfg.input.source,
            path=cfg.input.path,
            uniprot_id=cfg.input.uniprot_id,
            output_dir=out,
        )

        # 2. Clean
        cleaned = clean_structure(raw, out / "cleaned.pdb")

        # 3. Protonate
        protonated = protonate_structure(cleaned, out / "protonated.pdb", ph=cfg.input.ph)

        # 4. Apply phospho patches
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", str(protonated))

        manifest = []
        for site in cfg.modification.sites:
            modifier = get_modifier(site.phospho_type, cfg.forcefield)
            structure = modifier.apply(structure, chain_id=site.chain, resid=site.resid)
            manifest.append({
                "chain": site.chain,
                "resid": site.resid,
                "original_resname": site.resname,
                "phospho_type": site.phospho_type,
                "new_resname": modifier.new_resname,
            })
            logger.info("Applied %s to %s%d", site.phospho_type, site.chain, site.resid)

        # 5. Write outputs
        modified_pdb = out / "modified.pdb"
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(modified_pdb))

        manifest_path = out / "modification_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        return StageResult(
            stage="stage1",
            output_dir=out,
            artifacts={"modified_pdb": modified_pdb, "manifest": manifest_path},
        )
