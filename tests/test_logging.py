from __future__ import annotations
import logging
from pathlib import Path
from phosp.logging import configure_logging


def _clear_phosp_handlers() -> None:
    logging.getLogger("phosp").handlers.clear()


def test_configure_logging_attaches_stream_handler():
    _clear_phosp_handlers()
    configure_logging("WARNING")
    root = logging.getLogger("phosp")
    assert root.level == logging.WARNING
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], logging.StreamHandler)
    _clear_phosp_handlers()


def test_configure_logging_idempotent():
    _clear_phosp_handlers()
    configure_logging()
    configure_logging()
    assert len(logging.getLogger("phosp").handlers) == 1
    _clear_phosp_handlers()


def test_configure_logging_adds_file_handler(tmp_path: Path):
    _clear_phosp_handlers()
    log_file = tmp_path / "phosp.log"
    configure_logging(log_file=log_file)
    root = logging.getLogger("phosp")
    assert len(root.handlers) == 2
    handler_types = {type(h) for h in root.handlers}
    assert logging.StreamHandler in handler_types
    assert logging.FileHandler in handler_types
    _clear_phosp_handlers()
