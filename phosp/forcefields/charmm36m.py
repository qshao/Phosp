from __future__ import annotations
import logging
from pathlib import Path

from phosp.forcefields.base import ForceField

logger = logging.getLogger(__name__)

_PARAMS_DIR = Path(__file__).parent / "params" / "charmm36m"
_PHOSPHO_FILES = {"pSer": "sep.itp", "pThr": "tpo.itp", "pTyr": "ptr.itp"}


class CHARMM36mFF(ForceField):
    name = "charmm36m"

    def pdb2gmx_flag(self) -> str:
        return "charmm36m-jul2022"

    def get_phospho_params(self, phospho_type: str) -> Path:
        return _PARAMS_DIR / _PHOSPHO_FILES[phospho_type]

    def patch_topology(self, topology: Path, sites: list) -> Path:
        # CHARMM36m's forcefield.itp already contains SEP/TPO/PTR parameters;
        # pdb2gmx generates a complete topology without any extra ITP needed.
        return topology
