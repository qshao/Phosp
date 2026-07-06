from __future__ import annotations
import numpy as np
from phosp.modification.base import Modifier


class AcetylLysModifier(Modifier):
    """N-epsilon-acetyl-lysine (ALY): CHARMM36m already ships ALY in
    aminoacids.rtp, so no bundled parameter file is needed (same reasoning as
    the phospho patches)."""

    mod_type = "acetylLys"
    ff_resnames = {"charmm36m": "ALY"}

    def _build_atoms(self, residue) -> None:
        nz = residue["NZ"].get_vector().get_array()

        # CH (carbonyl C) placed along the CE->NZ direction extended past NZ —
        # approximate, same placement style as the phospho patches; relaxed by
        # the subsequent minimization MDP stage.
        direction = self._bond_direction(residue, "CE", "NZ", fallback=np.array([0.0, 0.0, 1.0]))
        ch_pos = nz + direction * 1.33  # amide C-N bond length
        self._add_atom(residue, "CH", ch_pos, "C")

        ref = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(direction, ref)) > 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        perp = np.cross(direction, ref)
        perp = perp / np.linalg.norm(perp)

        # OH (carbonyl O) and CH3 (methyl C), ~120 deg apart from NZ->CH in the
        # trigonal-planar amide arrangement around CH.
        oh_pos = ch_pos + (-0.5 * direction + 0.87 * perp) * 1.23   # C=O
        ch3_pos = ch_pos + (-0.5 * direction - 0.87 * perp) * 1.51  # C-C
        self._add_atom(residue, "OH", oh_pos, "O")
        self._add_atom(residue, "CH3", ch3_pos, "C")
