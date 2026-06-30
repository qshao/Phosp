from __future__ import annotations
import logging
from pathlib import Path

from phosp.forcefields.base import ForceField

logger = logging.getLogger(__name__)

_PARAMS_DIR = Path(__file__).parent / "params" / "amber_ff14sb"
_PHOSPHO_FILES = {"pSer": "sep.frcmod", "pThr": "tpo.frcmod", "pTyr": "ptr.frcmod"}


class AMBERff14SBFF(ForceField):
    name = "amber_ff14sb"

    def pdb2gmx_flag(self) -> str:
        return "amber99sb-ildn"

    def get_phospho_params(self, phospho_type: str) -> Path:
        return _PARAMS_DIR / _PHOSPHO_FILES[phospho_type]

    def patch_topology(self, topology: Path, sites: list) -> Path:
        content = topology.read_text()
        includes = []
        for site in sites:
            frcmod = self.get_phospho_params(site.phospho_type)
            include_line = f'#include "{frcmod}"\n'
            if include_line not in content:
                includes.append(include_line)
        if includes:
            topology.write_text("".join(includes) + content)
            logger.info("Patched topology with %d phospho includes", len(includes))
        return topology
