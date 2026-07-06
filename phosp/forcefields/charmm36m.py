from __future__ import annotations
import logging
import shutil
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

    def build_ncaa_forcefield(self, bundle_dirs: list[Path], base_ff_dir: Path, output_dir: Path) -> str:
        # GROMACS force-field directories merge *all* .rtp/.hdb files present,
        # not just aminoacids.rtp/aminoacids.hdb (confirmed: charmm36m-jul2022.ff
        # ships 9 separate .rtp files) — so new residues can live in their own
        # ncaa.rtp/ncaa.hdb without touching the shipped files.
        ext_name = f"{base_ff_dir.name.removesuffix('.ff')}-ncaa"
        ext_dir = output_dir / f"{ext_name}.ff"
        if ext_dir.exists():
            shutil.rmtree(ext_dir)
        shutil.copytree(base_ff_dir, ext_dir)

        rtp_chunks, hdb_chunks, itp_includes = [], [], []
        for bundle_dir in bundle_dirs:
            bundle_dir = Path(bundle_dir)
            rtp_chunks.append((bundle_dir / "residue.rtp").read_text())
            hdb_chunks.append((bundle_dir / "residue.hdb").read_text())
            params_itp = bundle_dir / "params.itp"
            if params_itp.exists():
                dest_name = f"{bundle_dir.name}_params.itp"
                shutil.copy2(params_itp, ext_dir / dest_name)
                itp_includes.append(dest_name)

        (ext_dir / "ncaa.rtp").write_text("\n\n".join(rtp_chunks) + "\n")
        (ext_dir / "ncaa.hdb").write_text("\n\n".join(hdb_chunks) + "\n")

        if itp_includes:
            ff_itp = ext_dir / "forcefield.itp"
            includes = "".join(f'#include "{name}"\n' for name in itp_includes)
            ff_itp.write_text(ff_itp.read_text() + "\n" + includes)

        logger.info("Built ncAA force field %s from %d bundle(s) at %s", ext_name, len(bundle_dirs), ext_dir)
        return ext_name
