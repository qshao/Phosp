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


class PhosphoSite(BaseModel):
    chain: str
    resid: int
    resname: Literal["SER", "THR", "TYR"]
    phospho_type: Literal["pSer", "pThr", "pTyr"]

    @model_validator(mode="after")
    def check_resname_phospho_type(self) -> PhosphoSite:
        mapping = {"SER": "pSer", "THR": "pThr", "TYR": "pTyr"}
        if mapping[self.resname] != self.phospho_type:
            raise ValueError(
                f"{self.resname} must use {mapping[self.resname]}, got {self.phospho_type}"
            )
        return self


class ModificationConfig(BaseModel):
    sites: list[PhosphoSite]


class HPCConfig(BaseModel):
    enabled: bool = False
    scheduler: Literal["slurm", "pbs"] = "slurm"
    ntasks: int = 8
    gpus: int = 1
    walltime: str = "24:00:00"
    partition: str = "gpu"
    auto_submit: bool = False

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
    gpu_id: int | None = None  # GPU index for mdrun; None = auto-detect
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
    rmsd: dict = Field(default_factory=lambda: {"selection": "backbone", "reference": "first_frame"})
    rmsf: dict = Field(default_factory=lambda: {"selection": "name CA"})
    mmpbsa: dict = Field(default_factory=lambda: {"method": "pbsa", "temperature": 300})
    sasa: dict = Field(default_factory=lambda: {"residues": []})


class PhospConfig(BaseModel):
    input: InputConfig
    modification: ModificationConfig
    forcefield: Literal["charmm36m", "amber_ff14sb"] = "charmm36m"
    engine: Literal["gromacs"] = "gromacs"
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
