from pathlib import Path
import pytest
from Bio.PDB import PDBParser
from phosp.modification.pser import PSerModifier
from phosp.modification.pthr import PThrModifier
from phosp.modification.ptyr import PTyrModifier
from phosp.modification.base import get_modifier

FIXTURES = Path(__file__).parent / "fixtures"


def _load(pdb_path):
    return PDBParser(QUIET=True).get_structure("p", str(pdb_path))


def test_pthr_renames_residue(tmp_path):
    struct = _load(FIXTURES / "ubiquitin.pdb")
    # Ubiquitin Thr 66 (chain A)
    mod = PThrModifier(forcefield="charmm36m")
    modified = mod.apply(struct, chain_id="A", resid=66)
    residues = {r.get_resname() for r in modified[0]["A"].get_residues()}
    assert "TPO" in residues


def test_pser_renames_residue(tmp_path):
    struct = _load(FIXTURES / "ubiquitin.pdb")
    mod = PSerModifier(forcefield="charmm36m")
    # Ubiquitin has no Ser in standard numbering; we'll use resid 20 (Asp) —
    # test checks rename behavior with a patched residue name instead
    # Use residue 57 (Ser in some Ubiquitin structures) if present, else skip
    ser_resids = [r.get_id()[1] for r in struct[0]["A"].get_residues() if r.get_resname() == "SER"]
    if not ser_resids:
        pytest.skip("No SER in fixture")
    modified = mod.apply(struct, chain_id="A", resid=ser_resids[0])
    residues = {r.get_resname() for r in modified[0]["A"].get_residues()}
    assert "SEP" in residues


def test_get_modifier_dispatch():
    mod = get_modifier("pSer", "charmm36m")
    assert isinstance(mod, PSerModifier)
    mod = get_modifier("pThr", "charmm36m")
    assert isinstance(mod, PThrModifier)
    mod = get_modifier("pTyr", "charmm36m")
    assert isinstance(mod, PTyrModifier)


def test_unknown_phospho_type_raises():
    with pytest.raises(ValueError, match="Unknown phospho_type"):
        get_modifier("pHis", "charmm36m")
