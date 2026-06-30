from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from phosp.utils.structure import protonate_structure


def test_protonate_structure_uses_configured_binary(tmp_path):
    """protonate_structure uses the pdb2pqr_binary kwarg, not hardcoded 'pdb2pqr'."""
    pdb = tmp_path / "in.pdb"
    pdb.write_text("ATOM ...")
    out = tmp_path / "out.pdb"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        protonate_structure(pdb, out, ph=7.4, pdb2pqr_binary="/opt/pdb2pqr/bin/pdb2pqr")

    called_cmd = mock_run.call_args.args[0]
    assert called_cmd[0] == "/opt/pdb2pqr/bin/pdb2pqr"


def test_protonate_structure_defaults_to_pdb2pqr(tmp_path):
    """protonate_structure defaults to 'pdb2pqr' when no binary specified."""
    pdb = tmp_path / "in.pdb"
    pdb.write_text("ATOM ...")
    out = tmp_path / "out.pdb"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        protonate_structure(pdb, out, ph=7.4)

    called_cmd = mock_run.call_args.args[0]
    assert called_cmd[0] == "pdb2pqr"
