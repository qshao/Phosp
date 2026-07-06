from __future__ import annotations
from abc import abstractmethod
import numpy as np
from phosp.modification.base import Modifier

# Phosphate atom names matching CHARMM36/AMBER RTP entries (P, O1P, O2P, O3P).
_P_NAME = "P"
_O_NAMES = ("O1P", "O2P", "O3P")


class PhosphoModifier(Modifier):
    """Shared tetrahedral phosphate-group geometry for pSer/pThr/pTyr."""

    @abstractmethod
    def _get_bridging_atom_name(self) -> str: ...

    def _build_atoms(self, residue) -> None:
        bridging_name = self._get_bridging_atom_name()
        bridging = residue[bridging_name].get_vector().get_array()

        # Place P along CB->bridging O direction, 1.61 Å from bridging O.
        direction = self._bond_direction(residue, "CB", bridging_name, fallback=np.array([0.0, 0.0, 1.0]))
        p_pos = bridging + direction * 1.61
        self._add_atom(residue, _P_NAME, p_pos, "P")

        # Three non-bridging oxygens in approximate tetrahedral arrangement around P.
        # Find first perpendicular vector (avoid collinearity with [0,1,0]).
        ref = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(direction, ref)) > 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        perp1 = np.cross(direction, ref)
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)

        offset = 1.52
        for name, vec in [
            (_O_NAMES[0], p_pos + perp1 * offset),
            (_O_NAMES[1], p_pos - perp1 * (offset * 0.5) + perp2 * (offset * 0.87)),
            (_O_NAMES[2], p_pos - perp1 * (offset * 0.5) - perp2 * (offset * 0.87)),
        ]:
            self._add_atom(residue, name, vec, "O")
