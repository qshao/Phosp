from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser, Structure
from Bio.PDB.Atom import Atom

logger = logging.getLogger(__name__)

# Atoms used to superpose the bundle's template.pdb onto the real backbone —
# these three (non-collinear) points fully determine the rigid-body transform.
_BACKBONE_ATOMS = ("N", "CA", "C")

# The only atoms kept from the *real* structure — every side-chain atom name
# (CB, CG, ...) follows the same Greek-letter convention across every amino
# acid regardless of actual chemistry, so a same-named atom already present
# on the target residue (e.g. the old side chain's own CB) is NOT the same
# atom as the template's CB; it must be discarded, not preserved by name.
_PRESERVED_ATOMS = frozenset({"N", "CA", "C", "O"})


def kabsch_fit(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation that best superpose mobile (N,3) onto target (N,3),
    i.e. target ~= mobile @ R.T + t. Standard SVD-based Kabsch algorithm."""
    mobile_centroid = mobile.mean(axis=0)
    target_centroid = target.mean(axis=0)
    mobile_c = mobile - mobile_centroid
    target_c = target - target_centroid

    H = mobile_c.T @ target_c
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    correction = np.diag([1.0, 1.0, d])
    R = Vt.T @ correction @ U.T
    t = target_centroid - R @ mobile_centroid
    return R, t


class NcaaModifier:
    """Grafts a noncanonical amino acid's atoms onto the real structure by
    superposing a user-supplied template.pdb's backbone onto the target site,
    rather than approximating atom placement via bond-vector geometry (ncAA
    side chains are too irregular for that — see PhosphoModifier/acetyl/methyl
    for the bond-vector approach used when it *is* applicable)."""

    def __init__(self, bundle_dir: Path, new_resname: str) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.new_resname = new_resname

    def apply(self, structure: Structure, chain_id: str, resid: int) -> Structure:
        residue = structure[0][chain_id][(" ", resid, " ")]

        template_path = self.bundle_dir / "template.pdb"
        template_struct = PDBParser(QUIET=True).get_structure("_ncaa_template", str(template_path))
        template_residue = next(template_struct[0].get_residues())

        mobile = np.array([template_residue[name].get_vector().get_array() for name in _BACKBONE_ATOMS])
        target = np.array([residue[name].get_vector().get_array() for name in _BACKBONE_ATOMS])
        R, t = kabsch_fit(mobile, target)

        # Strip everything but the preserved backbone from the real residue —
        # the template's side chain fully replaces the old one, not merges
        # with it by coincidental name match.
        for atom in list(residue.get_atoms()):
            if atom.get_name() not in _PRESERVED_ATOMS:
                residue.detach_child(atom.get_id())

        for atom in template_residue.get_atoms():
            name = atom.get_name()
            if name in _PRESERVED_ATOMS:
                continue
            new_coord = R @ atom.get_vector().get_array() + t
            element = atom.element.strip() if atom.element and atom.element.strip() else name[0]
            residue.add(Atom(name, new_coord, 0.0, 1.0, " ", name, 0, element))

        residue.resname = self.new_resname
        logger.info("Grafted ncAA bundle %s onto %s%d -> %s", self.bundle_dir, chain_id, resid, self.new_resname)
        return structure
