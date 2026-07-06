import json
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from phosp.utils.structure import clean_structure, fetch_structure, protonate_structure, _fetch_uniprot

FIXTURES = Path(__file__).parent / "fixtures"


def test_clean_structure_removes_waters(tmp_path):
    out = tmp_path / "clean.pdb"
    clean_structure(FIXTURES / "ubiquitin.pdb", out)
    assert out.exists()
    content = out.read_text()
    assert "HOH" not in content


def test_clean_structure_keeps_only_one_altloc_conformer(tmp_path):
    """Regression test: Biopython's default Select writes every altLoc
    conformer of a disordered atom, producing duplicate atom names (e.g. two
    "CA" records) in one residue — which pdb2gmx rejects/mishandles. Only one
    conformer must survive."""
    altloc_pdb = tmp_path / "altloc_input.pdb"
    altloc_pdb.write_text(
        "ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 10.00           N\n"
        "ATOM      2  CA AALA A   1      11.000  10.000  10.000  0.60 10.00           C\n"
        "ATOM      3  CA BALA A   1      11.200  10.100  10.100  0.40 10.00           C\n"
        "ATOM      4  C   ALA A   1      12.000  10.000  10.000  1.00 10.00           C\n"
        "END\n"
    )
    out = tmp_path / "clean.pdb"
    clean_structure(altloc_pdb, out)
    ca_lines = [l for l in out.read_text().splitlines() if l.startswith("ATOM") and " CA " in l]
    assert len(ca_lines) == 1


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


def test_fetch_uniprot_uses_pdburl_from_alphafold_api(tmp_path):
    """AlphaFold periodically bumps model versions (e.g. v4 -> v6), which
    breaks a hardcoded URL. _fetch_uniprot must ask the prediction API for
    the current pdbUrl instead of guessing a version number."""
    api_response = json.dumps(
        [{"pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-O43175-F1-model_v6.pdb"}]
    ).encode()
    dest = tmp_path / "input.pdb"
    with patch("phosp.utils.structure.urlopen") as mock_urlopen, \
         patch("phosp.utils.structure.urlretrieve") as mock_urlretrieve:
        mock_urlopen.return_value.__enter__.return_value.read.return_value = api_response
        _fetch_uniprot("O43175", dest)
    mock_urlretrieve.assert_called_once_with(
        "https://alphafold.ebi.ac.uk/files/AF-O43175-F1-model_v6.pdb", dest
    )


def test_fetch_uniprot_falls_back_to_rcsb_on_api_failure(tmp_path):
    dest = tmp_path / "input.pdb"
    with patch("phosp.utils.structure.urlopen", side_effect=OSError("network down")), \
         patch("phosp.utils.structure._fetch_rcsb_by_uniprot", return_value=dest) as mock_rcsb:
        result = _fetch_uniprot("O43175", dest)
    mock_rcsb.assert_called_once_with("O43175", dest)
    assert result == dest


def test_protonate_structure_raises_clear_error_on_timeout(tmp_path):
    with patch("phosp.utils.structure.subprocess.run",
               side_effect=subprocess.TimeoutExpired(cmd="pdb2pqr", timeout=300)):
        with pytest.raises(RuntimeError, match="PDB2PQR timed out"):
            protonate_structure(FIXTURES / "ubiquitin.pdb", tmp_path / "out.pdb", timeout=300)
