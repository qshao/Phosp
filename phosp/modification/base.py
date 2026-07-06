from __future__ import annotations
import logging
import numpy as np
from abc import ABC, abstractmethod
from Bio.PDB import Structure
from Bio.PDB.Atom import Atom

logger = logging.getLogger(__name__)

# mod_type -> "module.path.ClassName", lazily imported by get_modifier() so a
# single-site run doesn't pay the import cost of every modifier's dependencies.
# Adding a new PTM type is a new subclass module + one entry here.
_MODIFIER_REGISTRY: dict[str, str] = {
    "pSer": "phosp.modification.pser.PSerModifier",
    "pThr": "phosp.modification.pthr.PThrModifier",
    "pTyr": "phosp.modification.ptyr.PTyrModifier",
    "acetylLys": "phosp.modification.acetyl.AcetylLysModifier",
    "methylLys1": "phosp.modification.methyl.MethylLys1Modifier",
    "methylLys2": "phosp.modification.methyl.MethylLys2Modifier",
    "methylLys3": "phosp.modification.methyl.MethylLys3Modifier",
}


class Modifier(ABC):
    mod_type: str
    # Force field name -> patched residue name, e.g. {"charmm36m": "TPO"}.
    ff_resnames: dict[str, str]

    def __init__(self, forcefield: str) -> None:
        self.forcefield = forcefield
        try:
            self.new_resname = self.ff_resnames[forcefield]
        except KeyError:
            raise ValueError(
                f"{self.mod_type} does not support forcefield {forcefield!r}. "
                f"Supported: {sorted(self.ff_resnames)}"
            ) from None

    @abstractmethod
    def _build_atoms(self, residue) -> None:
        """Add/rename the heavy atoms that turn the source residue into new_resname.

        Hydrogens are not this method's concern: pdb2gmx is invoked with -ignh
        and regenerates them from new_resname's .hdb entry regardless of what
        hydrogens are present in the input structure.
        """

    def apply(self, structure: Structure, chain_id: str, resid: int) -> Structure:
        residue = structure[0][chain_id][(" ", resid, " ")]
        residue.resname = self.new_resname
        self._build_atoms(residue)
        logger.info("Patched %s %s%d -> %s", self.mod_type, chain_id, resid, self.new_resname)
        return structure

    @staticmethod
    def _bond_direction(residue, from_name: str, to_name: str, fallback: np.ndarray) -> np.ndarray:
        """Unit vector from atom from_name to atom to_name; fallback if from_name is absent."""
        to_pos = residue[to_name].get_vector().get_array()
        try:
            from_pos = residue[from_name].get_vector().get_array()
        except KeyError:
            return fallback / np.linalg.norm(fallback)

        direction = to_pos - from_pos
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            return fallback / np.linalg.norm(fallback)
        return direction / norm

    @staticmethod
    def _add_atom(residue, name: str, coord: np.ndarray, element: str) -> None:
        if name in [a.get_name() for a in residue.get_atoms()]:
            return
        atom = Atom(name, coord, 0.0, 1.0, " ", name, 0, element)
        residue.add(atom)


def get_modifier(mod_type: str, forcefield: str) -> Modifier:
    import importlib

    path = _MODIFIER_REGISTRY.get(mod_type)
    if path is None:
        raise ValueError(f"Unknown mod_type: {mod_type}")
    module_path, cls_name = path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls(forcefield)
