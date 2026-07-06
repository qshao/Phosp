from pathlib import Path
import pytest
from phosp.forcefields.charmm36m import CHARMM36mFF
from phosp.forcefields.amber_ff14sb import AMBERff14SBFF


def test_charmm36m_pdb2gmx_flag():
    ff = CHARMM36mFF()
    assert "charmm36" in ff.pdb2gmx_flag()


def test_charmm36m_modification_params_exist():
    ff = CHARMM36mFF()
    for pt in ["pSer", "pThr", "pTyr"]:
        p = ff.get_modification_params(pt)
        assert p.exists(), f"Missing param file for {pt}: {p}"


def test_charmm36m_unknown_mod_type():
    ff = CHARMM36mFF()
    with pytest.raises(KeyError):
        ff.get_modification_params("pHis")


def test_amber_pdb2gmx_flag():
    ff = AMBERff14SBFF()
    assert "amber" in ff.pdb2gmx_flag()


def test_amber_modification_params_exist():
    ff = AMBERff14SBFF()
    for pt in ["pSer", "pThr", "pTyr"]:
        p = ff.get_modification_params(pt)
        assert p.exists(), f"Missing param file for {pt}: {p}"
