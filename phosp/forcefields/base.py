from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path


class ForceField(ABC):
    name: str

    @abstractmethod
    def get_modification_params(self, mod_type: str) -> Path:
        """Return path to bundled parameter file for a modification type, if any."""

    @abstractmethod
    def patch_topology(self, topology: Path, sites: list) -> Path:
        """Merge phospho-residue parameters into topology; return updated topology path."""

    @abstractmethod
    def pdb2gmx_flag(self) -> str:
        """Return the -ff argument value for gmx pdb2gmx, e.g. 'charmm36m-jul2022'."""

    def build_ncaa_forcefield(self, bundle_dirs: list[Path], base_ff_dir: Path, output_dir: Path) -> str:
        """Build a per-run force-field directory extended with ncAA residue
        bundles, returning the -ff argument value to use with it. Not every
        force field supports this (see CHARMM36mFF for the only current
        implementation)."""
        raise NotImplementedError(
            f"{self.name} does not support noncanonical amino acids. "
            "Use forcefield: charmm36m instead."
        )
