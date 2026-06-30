from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StageResult:
    stage: str
    output_dir: Path
    artifacts: dict[str, Path] = field(default_factory=dict)


class Stage(ABC):
    def __init__(self, config, engine, forcefield, output_root: Path) -> None:
        self.config = config
        self.engine = engine
        self.forcefield = forcefield
        self.output_root = output_root

    @abstractmethod
    def validate_inputs(self) -> None:
        """Raise StageInputError if preconditions are not met."""

    @abstractmethod
    def run(self) -> StageResult:
        """Execute the stage and return paths to produced artifacts."""
