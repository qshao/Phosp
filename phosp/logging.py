from __future__ import annotations
import logging
from pathlib import Path

_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    root = logging.getLogger("phosp")
    level_upper = level.upper()
    if level_upper not in _VALID_LEVELS:
        raise ValueError(f"Invalid log level {level!r}. Choose from: {sorted(_VALID_LEVELS)}")
    root.setLevel(getattr(logging, level_upper))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    if log_file:
        resolved = str(Path(log_file).resolve())
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == resolved
                   for h in root.handlers):
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            root.addHandler(fh)
