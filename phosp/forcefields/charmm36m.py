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
        content = topology.read_text()
        includes = []
        for site in sites:
            itp = self.get_phospho_params(site.phospho_type)
            include_line = f'#include "{itp}"\n'
            if include_line not in content:
                includes.append(include_line)
        if includes:
            insert_after = "; Include Position restraint file"
            if insert_after in content:
                content = content.replace(
                    insert_after, "".join(includes) + insert_after, 1
                )
            else:
                content = "".join(includes) + content
            topology.write_text(content)
            logger.info("Patched topology with %d phospho includes", len(includes))
        return topology
