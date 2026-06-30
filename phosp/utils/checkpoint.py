from __future__ import annotations
import json
import os
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

    def _write(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2))
        os.replace(tmp, self.path)

    def mark_stage_started(self, stage: str) -> None:
        self._data[f"{stage}_started_at"] = datetime.now().isoformat()
        self._write()

    def get_duration(self, stage: str) -> float | None:
        return self._data.get(f"{stage}_duration_seconds")

    def mark_complete(
        self,
        stage: str,
        artifacts: dict[str, str],
        config_hash: str | None = None,
    ) -> None:
        if stage not in self._data["completed_stages"]:
            self._data["completed_stages"].append(stage)
        self._data["artifacts"][stage] = artifacts
        self._data[f"{stage}_completed_at"] = datetime.now().isoformat()
        started_at_str = self._data.get(f"{stage}_started_at")
        if started_at_str:
            try:
                started = datetime.fromisoformat(started_at_str)
                duration = (datetime.now() - started).total_seconds()
                self._data[f"{stage}_duration_seconds"] = duration
            except (ValueError, TypeError):
                pass
        if config_hash is not None and "config_hash" not in self._data:
            self._data["config_hash"] = config_hash
        self._write()

    def is_complete(self, stage: str) -> bool:
        return stage in self._data["completed_stages"]

    def get_artifacts(self, stage: str) -> dict[str, str]:
        return self._data["artifacts"].get(stage, {})

    def get_config_hash(self) -> str | None:
        return self._data.get("config_hash")

    def store_config_hash(self, hash_value: str) -> None:
        if "config_hash" not in self._data:
            self._data["config_hash"] = hash_value
            self._write()
