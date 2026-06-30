from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path


class Checkpoint:
    def __init__(self, checkpoint_file: Path) -> None:
        self.path = checkpoint_file
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {"completed_stages": [], "artifacts": {}}

    def mark_complete(self, stage: str, artifacts: dict[str, str]) -> None:
        if stage not in self._data["completed_stages"]:
            self._data["completed_stages"].append(stage)
        self._data["artifacts"][stage] = artifacts
        self._data[f"{stage}_completed_at"] = datetime.now().isoformat()
        self.path.write_text(json.dumps(self._data, indent=2))

    def is_complete(self, stage: str) -> bool:
        return stage in self._data["completed_stages"]

    def get_artifacts(self, stage: str) -> dict[str, str]:
        return self._data["artifacts"].get(stage, {})
