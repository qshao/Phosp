from __future__ import annotations
import os
import re
import subprocess
from pathlib import Path


def discover_top_dir(gromacs_binary: str, ff_dirname: str) -> Path | None:
    """Locate the GROMACS force-field search directory that would contain
    ff_dirname (e.g. 'charmm36m-jul2022.ff') — preferring GMXLIB (GROMACS's
    own mechanism for adding a search path without write access to the
    binary's install prefix) over the install's own Data prefix. Returns the
    top_dir even if ff_dirname doesn't exist inside it yet, so callers can
    still produce a helpful "not found, install it here" message; returns
    None only if no candidate top_dir could be determined at all."""
    gmxlib = os.environ.get("GMXLIB")
    if gmxlib and (Path(gmxlib) / ff_dirname).exists():
        return Path(gmxlib)

    try:
        out = subprocess.run([gromacs_binary, "--version"], capture_output=True, text=True).stdout
        m = re.search(r"Data prefix:\s*(.+)", out)
        if not m:
            return None
        return Path(m.group(1).strip()) / "share" / "gromacs" / "top"
    except Exception:
        return None
