from __future__ import annotations
import logging
import numpy as np
from abc import ABC, abstractmethod
from Bio.PDB import Structure
from Bio.PDB.Atom import Atom

logger = logging.getLogger(__name__)

# FF-specific residue names for phosphorylated residues
_FF_NAMES: dict[str, dict[str, str]] = {
    "charmm36m": {"pSer": "SEP", "pThr": "TPO", "pTyr": "PTR"},
    "amber_ff14sb": {"pSer": "SEP", "pThr": "TPO", "pTyr": "PTR"},
}

# Phosphate atom names added to residue (CHARMM/AMBER convention)
_PHOSPHO_ATOMS = {
    "pSer": {"bridging_O": "OG",  "P": "PG",  "O1": "O1G", "O2": "O2G", "O3": "O3G"},
    "pThr": {"bridging_O": "OG1", "P": "PG",  "O1": "O1G", "O2": "O2G", "O3": "O3G"},
    "pTyr": {"bridging_O": "OH",  "P": "PH",  "O1": "O1H", "O2": "O2H", "O3": "O3H"},
}


class Modifier(ABC):
    phospho_type: str

    def __init__(self, forcefield: str) -> None:
        self.forcefield = forcefield
        self.new_resname = _FF_NAMES[forcefield][self.phospho_type]

    @abstractmethod
    def _get_bridging_atom_name(self) -> str: ...

    def apply(self, structure: Structure, chain_id: str, resid: int) -> Structure:
        residue = structure[0][chain_id][(" ", resid, " ")]
        residue.resname = self.new_resname
        self._add_phosphate_atoms(residue)
        logger.info("Patched %s %s%d -> %s", self.phospho_type, chain_id, resid, self.new_resname)
        return structure

    def _add_phosphate_atoms(self, residue) -> None:
        atom_names = _PHOSPHO_ATOMS[self.phospho_type]
        bridging = residue[atom_names["bridging_O"]].get_vector().get_array()

        # Place P along CB->bridging O direction, 1.61 Å from bridging O
        try:
            cb = residue["CB"].get_vector().get_array()
        except KeyError:
            cb = bridging + np.array([0.0, 0.0, -1.0])

        direction = bridging - cb
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            direction = np.array([0.0, 0.0, 1.0])
        else:
            direction = direction / norm

        p_pos = bridging + direction * 1.61

        self._add_atom(residue, atom_names["P"], p_pos, "P")

        # Three non-bridging oxygens in approximate tetrahedral arrangement around P
        # Find first perpendicular vector (avoid collinearity with [0,1,0])
        ref = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(direction, ref)) > 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        perp1 = np.cross(direction, ref)
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)

        offset = 1.52
        for name, vec in [
            (atom_names["O1"], p_pos + perp1 * offset),
            (atom_names["O2"], p_pos - perp1 * (offset * 0.5) + perp2 * (offset * 0.87)),
            (atom_names["O3"], p_pos - perp1 * (offset * 0.5) - perp2 * (offset * 0.87)),
        ]:
            self._add_atom(residue, name, vec, "O")

    @staticmethod
    def _add_atom(residue, name: str, coord: np.ndarray, element: str) -> None:
        if name in [a.get_name() for a in residue.get_atoms()]:
            return
        atom = Atom(name, coord, 0.0, 1.0, " ", name, 0, element)
        residue.add(atom)


def get_modifier(phospho_type: str, forcefield: str) -> Modifier:
    from phosp.modification.pser import PSerModifier
    from phosp.modification.pthr import PThrModifier
    from phosp.modification.ptyr import PTyrModifier
    match phospho_type:
        case "pSer":
            return PSerModifier(forcefield)
        case "pThr":
            return PThrModifier(forcefield)
        case "pTyr":
            return PTyrModifier(forcefield)
        case _:
            raise ValueError(f"Unknown phospho_type: {phospho_type}")
