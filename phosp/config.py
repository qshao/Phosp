from __future__ import annotations
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class InputConfig(BaseModel):
    source: Literal["pdb", "uniprot"]
    path: Path | None = None
    uniprot_id: str | None = None
    ph: float = 7.4

    @model_validator(mode="after")
    def check_source_fields(self) -> InputConfig:
        if self.source == "pdb" and not self.path:
            raise ValueError("path required when source=pdb")
        if self.source == "uniprot" and not self.uniprot_id:
            raise ValueError("uniprot_id required when source=uniprot")
        return self


# Maps each supported modification type to the source residue it applies to.
# Kept here (rather than importing phosp.modification, which pulls in Bio.PDB/
# numpy) since config.py is on the hot path for every CLI invocation. Adding a
# new Modifier subclass requires a matching one-line entry here.
_SOURCE_RESNAME: dict[str, str] = {
    "pSer": "SER",
    "pThr": "THR",
    "pTyr": "TYR",
    "acetylLys": "LYS",
    "methylLys1": "LYS",
    "methylLys2": "LYS",
    "methylLys3": "LYS",
}


class ModificationSite(BaseModel):
    chain: str
    resid: int
    resname: str
    mod_type: str

    @model_validator(mode="after")
    def check_resname_mod_type(self) -> ModificationSite:
        if self.mod_type not in _SOURCE_RESNAME:
            raise ValueError(
                f"Unknown mod_type: {self.mod_type!r}. "
                f"Known types: {sorted(_SOURCE_RESNAME)}"
            )
        expected = _SOURCE_RESNAME[self.mod_type]
        if self.resname != expected:
            raise ValueError(
                f"{self.mod_type} requires resname {expected}, got {self.resname}"
            )
        return self


# Residue codes already meaningful to the bundled force fields — a user-chosen
# ncAA new_resname must not collide with any of these or pdb2gmx would pick up
# the wrong (or an ambiguous) residue definition.
_RESERVED_RESNAMES = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "SEP", "TPO", "PTR", "ALY", "MLZ", "MLY", "M3L",
})


class NcaaSite(BaseModel):
    """A noncanonical amino acid site backed by a user-supplied parameter
    bundle (see the ncAA plan) — kept separate from ModificationSite because
    new_resname/bundle_dir are open-ended, not validated against a closed
    registry the way mod_type is."""

    chain: str
    resid: int
    resname: str       # source residue being replaced, e.g. "MET"
    new_resname: str   # user-chosen 3-letter code for the bundle's residue
    bundle_dir: Path    # directory containing residue.rtp/residue.hdb/template.pdb[/params.itp]

    @model_validator(mode="after")
    def check_new_resname(self) -> NcaaSite:
        if self.new_resname in _RESERVED_RESNAMES:
            raise ValueError(
                f"new_resname {self.new_resname!r} collides with an existing "
                f"force-field residue code; choose a distinct 3-letter code"
            )
        if not self.bundle_dir.is_dir():
            raise ValueError(f"bundle_dir not found: {self.bundle_dir}")
        return self


class ModificationConfig(BaseModel):
    sites: list[ModificationSite] = Field(default_factory=list)
    ncaa_sites: list[NcaaSite] = Field(default_factory=list)


class HPCConfig(BaseModel):
    """Resource settings used by the slurm and pbs runners."""
    ntasks: int = 8          # OpenMP threads per rank (cpus-per-task for SLURM)
    gpus: int = 1
    walltime: str = "24:00:00"
    partition: str = "gpu"
    auto_submit: bool = False
    gromacs_module: str | None = None  # e.g. "gromacs/2026.0-cuda"; None = skip module load
    extra_directives: list[str] = Field(default_factory=list)
    # Arbitrary scheduler directives appended verbatim after the standard headers.
    # SLURM: ["--account=myproject", "--qos=high", "--constraint=a100", "--mem=128G"]
    # PBS:   ["-l mem=128gb", "-A myproject"]

    @field_validator("ntasks")
    @classmethod
    def ntasks_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("ntasks must be >= 1")
        return v

    @field_validator("gpus")
    @classmethod
    def gpus_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("gpus must be >= 0")
        return v


class SimulationConfig(BaseModel):
    production_time_ns: float = 100.0
    output_freq_ps: float = 10.0
    water_model: Literal["tip3p", "spce"] = "tip3p"
    box_type: Literal["dodecahedron", "cubic"] = "dodecahedron"
    salt_concentration_mM: float = 150.0
    gpu_id: int | None = None  # GPU index for mdrun; None = let GROMACS auto-select
    runner: Literal["local", "slurm", "pbs"] = "local"
    hpc: HPCConfig = Field(default_factory=HPCConfig)

    @field_validator("production_time_ns")
    @classmethod
    def production_time_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("production_time_ns must be > 0")
        return v

    @field_validator("output_freq_ps")
    @classmethod
    def output_freq_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("output_freq_ps must be > 0")
        return v

    @field_validator("salt_concentration_mM")
    @classmethod
    def salt_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("salt_concentration_mM must be >= 0")
        return v

    @model_validator(mode="after")
    def output_freq_fits_in_production(self) -> SimulationConfig:
        if self.output_freq_ps > self.production_time_ns * 1000:
            raise ValueError(
                f"output_freq_ps ({self.output_freq_ps} ps) exceeds "
                f"production_time_ns ({self.production_time_ns} ns = "
                f"{self.production_time_ns * 1000} ps) — no frames would be written"
            )
        return self


class AnalysisConfig(BaseModel):
    model_config = {"extra": "allow"}
    plugins: list[str] = Field(default_factory=list)
    rmsd: dict = Field(default_factory=lambda: {"selection": "name CA"})
    rmsf: dict = Field(default_factory=lambda: {"selection": "name CA"})
    mmpbsa: dict = Field(default_factory=lambda: {"method": "pbsa", "temperature": 300})
    sasa: dict = Field(default_factory=lambda: {"residues": []})


class GROMACSConfig(BaseModel):
    binary: str = "gmx"  # path or name of the gmx binary (e.g. "gmx", "gmx_mpi", "/opt/gromacs/bin/gmx")
    timeout_minutes: int | None = None  # None = no limit; e.g. 120 for 2-hour hard cap
    pdb2pqr: str = "pdb2pqr"  # name or full path, e.g. "/opt/pdb2pqr/bin/pdb2pqr"

    @field_validator("timeout_minutes")
    @classmethod
    def timeout_minutes_positive(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("timeout_minutes must be > 0 (or omitted for no limit)")
        return v


class PhospConfig(BaseModel):
    input: InputConfig
    modification: ModificationConfig
    forcefield: Literal["charmm36m", "amber_ff14sb"] = "charmm36m"
    engine: Literal["gromacs"] = "gromacs"
    gromacs: GROMACSConfig = Field(default_factory=GROMACSConfig)
    protocol: str = "globular_protein"
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)


def load_config(path: Path) -> PhospConfig:
    from pydantic import ValidationError
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    try:
        return PhospConfig.model_validate(data)
    except ValidationError as exc:
        lines = ["Config validation failed:"]
        for err in exc.errors():
            loc = ".".join(str(x) for x in err["loc"]) if err["loc"] else "(root)"
            lines.append(f"  {loc}: {err['msg']}")
        raise ValueError("\n".join(lines)) from None
