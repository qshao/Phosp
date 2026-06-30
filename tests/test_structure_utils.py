import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from phosp.utils.structure import clean_structure, fetch_structure, protonate_structure

FIXTURES = Path(__file__).parent / "fixtures"


def test_clean_structure_removes_waters(tmp_path):
    out = tmp_path / "clean.pdb"
    clean_structure(FIXTURES / "ubiquitin.pdb", out)
    assert out.exists()
    content = out.read_text()
    assert "HOH" not in content


def test_clean_structure_keeps_hetatm_when_specified(tmp_path):
    # 1UBQ has no ligands but we can verify the keep_hetatm param is respected
    out = tmp_path / "clean.pdb"
    clean_structure(FIXTURES / "ubiquitin.pdb", out, keep_hetatm=["ZN"])
    assert out.exists()


def test_fetch_structure_pdb_source_copies_file(tmp_path):
    result = fetch_structure(
        source="pdb",
        path=FIXTURES / "ubiquitin.pdb",
        uniprot_id=None,
        output_dir=tmp_path,
    )
    assert result.exists()
    assert result.name == "input.pdb"


def test_protonate_structure_passes_timeout_to_subprocess(tmp_path):
    with patch("phosp.utils.structure.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        protonate_structure(FIXTURES / "ubiquitin.pdb", tmp_path / "out.pdb", timeout=300)
    assert mock_run.call_args.kwargs["timeout"] == 300


def test_protonate_structure_raises_clear_error_on_timeout(tmp_path):
    with patch("phosp.utils.structure.subprocess.run",
               side_effect=subprocess.TimeoutExpired(cmd="pdb2pqr", timeout=300)):
        with pytest.raises(RuntimeError, match="PDB2PQR timed out"):
            protonate_structure(FIXTURES / "ubiquitin.pdb", tmp_path / "out.pdb", timeout=300)
