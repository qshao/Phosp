from __future__ import annotations
import json
import logging
import shutil
from pathlib import Path

from Bio.PDB import PDBIO

from phosp.exceptions import PhospError, StageInputError
from phosp.modification.base import get_modifier
from phosp.modification.ncaa import NcaaModifier
from phosp.stages.base import Stage, StageResult
from phosp.utils.structure import clean_structure, fetch_structure, protonate_structure

logger = logging.getLogger(__name__)


class Stage1Modify(Stage):
    def __init__(self, config, engine, forcefield, output_root: Path, ui=None, reference_mode: bool = False) -> None:
        super().__init__(config, engine, forcefield, output_root, ui)
        self.reference_mode = reference_mode

    def validate_inputs(self) -> None:
        src = self.config.input
        if src.source == "pdb" and not src.path.exists():
            raise StageInputError(f"Input PDB not found: {src.path}")
        pdb2pqr = self.config.gromacs.pdb2pqr
        if shutil.which(pdb2pqr) is None and not Path(pdb2pqr).is_file():
            raise PhospError(
                f"pdb2pqr binary '{pdb2pqr}' not found. "
                "Install it with 'pip install pdb2pqr' or set gromacs.pdb2pqr in your config."
            )
        if not self.reference_mode and src.source == "pdb" and src.path.exists():
            from Bio.PDB import PDBParser
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("_", str(src.path))
            present = {
                (chain.id, res.id[1])
                for model in structure
                for chain in model
                for res in chain
                if res.id[0] == " "
            }
            for site in self.config.modification.sites:
                if (site.chain, site.resid) not in present:
                    raise StageInputError(
                        f"Modification site chain {site.chain} resid {site.resid} "
                        f"not found in {src.path}. "
                        "Check the chain ID and residue number in your config."
                    )
            for site in self.config.modification.ncaa_sites:
                if (site.chain, site.resid) not in present:
                    raise StageInputError(
                        f"ncAA site chain {site.chain} resid {site.resid} "
                        f"not found in {src.path}. "
                        "Check the chain ID and residue number in your config."
                    )

    def run(self) -> StageResult:
        out = self.output_root
        out.mkdir(parents=True, exist_ok=True)
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
        timeout_minutes = cfg.gromacs.timeout_minutes
        protonated = protonate_structure(
            cleaned, out / "protonated.pdb",
            ph=cfg.input.ph,
            pdb2pqr_binary=cfg.gromacs.pdb2pqr,
            timeout=timeout_minutes * 60 if timeout_minutes else None,
        )

        if self.reference_mode:
            # Reference run: use protonated structure as-is (no phospho patches)
            modified_pdb = out / "modified.pdb"
            shutil.copy2(protonated, modified_pdb)
            logger.info("Reference mode: skipping phosphorylation, using protonated structure as-is")
            manifest = []
        else:
            # 4. Apply phospho patches
            from Bio.PDB import PDBParser
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", str(protonated))

            manifest = []
            for site in cfg.modification.sites:
                modifier = get_modifier(site.mod_type, cfg.forcefield)
                structure = modifier.apply(structure, chain_id=site.chain, resid=site.resid)
                manifest.append({
                    "kind": "ptm",
                    "chain": site.chain,
                    "resid": site.resid,
                    "original_resname": site.resname,
                    "mod_type": site.mod_type,
                    "new_resname": modifier.new_resname,
                })
                logger.info("Applied %s to %s%d", site.mod_type, site.chain, site.resid)

            for site in cfg.modification.ncaa_sites:
                ncaa_modifier = NcaaModifier(site.bundle_dir, site.new_resname)
                structure = ncaa_modifier.apply(structure, chain_id=site.chain, resid=site.resid)
                manifest.append({
                    "kind": "ncaa",
                    "chain": site.chain,
                    "resid": site.resid,
                    "original_resname": site.resname,
                    "new_resname": site.new_resname,
                    "bundle_dir": str(site.bundle_dir),
                })
                logger.info("Applied ncAA bundle %s to %s%d", site.bundle_dir, site.chain, site.resid)

            # 5. Write modified structure
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
