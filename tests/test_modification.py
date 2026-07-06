from pathlib import Path
import pytest
from Bio.PDB import PDBParser
from phosp.modification.pser import PSerModifier
from phosp.modification.pthr import PThrModifier
from phosp.modification.ptyr import PTyrModifier
from phosp.modification.acetyl import AcetylLysModifier
from phosp.modification.methyl import MethylLys1Modifier, MethylLys2Modifier, MethylLys3Modifier
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


def _first_lys_resid(struct):
    lys_resids = [r.get_id()[1] for r in struct[0]["A"].get_residues() if r.get_resname() == "LYS"]
    if not lys_resids:
        pytest.skip("No LYS in fixture")
    return lys_resids[0]


def test_acetyllys_renames_residue(tmp_path):
    struct = _load(FIXTURES / "ubiquitin.pdb")
    resid = _first_lys_resid(struct)
    mod = AcetylLysModifier(forcefield="charmm36m")
    modified = mod.apply(struct, chain_id="A", resid=resid)
    residues = {r.get_resname() for r in modified[0]["A"].get_residues()}
    assert "ALY" in residues


@pytest.mark.parametrize("modifier_cls,expected_resname", [
    (MethylLys1Modifier, "MLZ"),
    (MethylLys2Modifier, "MLY"),
    (MethylLys3Modifier, "M3L"),
])
def test_methyllys_renames_residue(tmp_path, modifier_cls, expected_resname):
    struct = _load(FIXTURES / "ubiquitin.pdb")
    resid = _first_lys_resid(struct)
    mod = modifier_cls(forcefield="charmm36m")
    modified = mod.apply(struct, chain_id="A", resid=resid)
    residues = {r.get_resname() for r in modified[0]["A"].get_residues()}
    assert expected_resname in residues


def test_get_modifier_dispatch():
    mod = get_modifier("pSer", "charmm36m")
    assert isinstance(mod, PSerModifier)
    mod = get_modifier("pThr", "charmm36m")
    assert isinstance(mod, PThrModifier)
    mod = get_modifier("pTyr", "charmm36m")
    assert isinstance(mod, PTyrModifier)
    mod = get_modifier("acetylLys", "charmm36m")
    assert isinstance(mod, AcetylLysModifier)
    mod = get_modifier("methylLys2", "charmm36m")
    assert isinstance(mod, MethylLys2Modifier)


def test_unknown_mod_type_raises():
    with pytest.raises(ValueError, match="Unknown mod_type"):
        get_modifier("pHis", "charmm36m")


def test_unsupported_forcefield_raises_clean_error():
    """Regression test: acetylLys/methylLys* only have a charmm36m entry in
    ff_resnames — requesting amber_ff14sb must raise a clear ValueError, not
    an undocumented bare KeyError from the ff_resnames[forcefield] lookup."""
    with pytest.raises(ValueError, match="acetylLys does not support forcefield"):
        get_modifier("acetylLys", "amber_ff14sb")
