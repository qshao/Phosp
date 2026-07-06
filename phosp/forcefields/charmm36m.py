from __future__ import annotations
import logging
from pathlib import Path

from phosp.forcefields.base import ForceField

logger = logging.getLogger(__name__)

_PARAMS_DIR = Path(__file__).parent / "params" / "charmm36m"
# CHARMM36m's own aminoacids.rtp already ships parameters for every
# modification type below (SEP/TPO/PTR, and — verified against the installed
# charmm36m-jul2022.ff — ALY/MLZ/MLY/M3L too), so pdb2gmx generates a complete
# topology without any extra ITP for most of them; these bundled files are
# only consulted where callers explicitly ask for one.
_MOD_FILES = {"pSer": "sep.itp", "pThr": "tpo.itp", "pTyr": "ptr.itp"}


class CHARMM36mFF(ForceField):
    name = "charmm36m"

    def pdb2gmx_flag(self) -> str:
        return "charmm36m-jul2022"

    def get_modification_params(self, mod_type: str) -> Path:
        return _PARAMS_DIR / _MOD_FILES[mod_type]

    def patch_topology(self, topology: Path, sites: list) -> Path:
        # CHARMM36m's forcefield.itp already contains parameters for every
        # modification type this pipeline supports; pdb2gmx generates a
        # complete topology without any extra ITP needed.
        return topology
