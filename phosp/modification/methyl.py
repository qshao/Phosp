from __future__ import annotations
import numpy as np
from phosp.modification.base import Modifier

# CHARMM36m aminoacids.rtp ships MLZ/MLY/M3L (mono/di/tri-methyl-lysine)
# natively — same reasoning as the phospho patches, no bundled parameter file
# needed. Atom names below are taken directly from that rtp (they aren't
# consistent across the three residues — MLY uses CH1/CH2, M3L uses
# CM1/CM2/CM3 — so each class spells them out rather than deriving a pattern).
_CN_BOND = 1.51  # approximate amine/ammonium C-N bond length (Å)


def _perp_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(direction, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    perp1 = np.cross(direction, ref)
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)
    return perp1, perp2


class MethylLys1Modifier(Modifier):
    """Mono-methyl-lysine (MLZ): adds CM to NZ."""

    mod_type = "methylLys1"
    ff_resnames = {"charmm36m": "MLZ"}

    def _build_atoms(self, residue) -> None:
        nz = residue["NZ"].get_vector().get_array()
        direction = self._bond_direction(residue, "CE", "NZ", fallback=np.array([0.0, 0.0, 1.0]))
        perp1, _ = _perp_basis(direction)
        self._add_atom(residue, "CM", nz + perp1 * _CN_BOND, "C")


class MethylLys2Modifier(Modifier):
    """Di-methyl-lysine (MLY): adds CH1, CH2 to NZ."""

    mod_type = "methylLys2"
    ff_resnames = {"charmm36m": "MLY"}

    def _build_atoms(self, residue) -> None:
        nz = residue["NZ"].get_vector().get_array()
        direction = self._bond_direction(residue, "CE", "NZ", fallback=np.array([0.0, 0.0, 1.0]))
        perp1, perp2 = _perp_basis(direction)
        self._add_atom(residue, "CH1", nz + perp1 * _CN_BOND, "C")
        self._add_atom(residue, "CH2", nz + (-0.5 * perp1 + 0.87 * perp2) * _CN_BOND, "C")


class MethylLys3Modifier(Modifier):
    """Tri-methyl-lysine (M3L): adds CM1, CM2, CM3 to NZ (quaternary ammonium)."""

    mod_type = "methylLys3"
    ff_resnames = {"charmm36m": "M3L"}

    def _build_atoms(self, residue) -> None:
        nz = residue["NZ"].get_vector().get_array()
        direction = self._bond_direction(residue, "CE", "NZ", fallback=np.array([0.0, 0.0, 1.0]))
        perp1, perp2 = _perp_basis(direction)
        for name, vec in [
            ("CM1", nz + perp1 * _CN_BOND),
            ("CM2", nz + (-0.5 * perp1 + 0.87 * perp2) * _CN_BOND),
            ("CM3", nz + (-0.5 * perp1 - 0.87 * perp2) * _CN_BOND),
        ]:
            self._add_atom(residue, name, vec, "C")
