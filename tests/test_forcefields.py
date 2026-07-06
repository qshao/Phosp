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


def test_amber_ncaa_not_supported():
    ff = AMBERff14SBFF()
    with pytest.raises(NotImplementedError, match="does not support"):
        ff.build_ncaa_forcefield([], Path("unused"), Path("unused"))


def _make_fake_base_ff(tmp_path: Path) -> Path:
    ff_dir = tmp_path / "fake-jul2022.ff"
    ff_dir.mkdir()
    (ff_dir / "aminoacids.rtp").write_text("[ ALA ]\n  [ atoms ]\n")
    (ff_dir / "forcefield.itp").write_text(
        '#define _FF_FAKE\n#include "ffnonbonded.itp"\n#include "ffbonded.itp"\n'
    )
    return ff_dir


def test_build_ncaa_forcefield_merges_bundle_rtp_hdb(tmp_path):
    base_ff = _make_fake_base_ff(tmp_path)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "residue.rtp").write_text("[ XAA ]\n  [ atoms ]\n        N      NH1 -0.5000   1\n")
    (bundle / "residue.hdb").write_text("XAA        0\n")

    ff = CHARMM36mFF()
    output_dir = tmp_path / "stage2"
    output_dir.mkdir()
    ff_flag = ff.build_ncaa_forcefield([bundle], base_ff, output_dir)

    assert ff_flag == "fake-jul2022-ncaa"
    ext_dir = output_dir / f"{ff_flag}.ff"
    assert ext_dir.is_dir()
    assert "XAA" in (ext_dir / "ncaa.rtp").read_text()
    assert "XAA" in (ext_dir / "ncaa.hdb").read_text()
    # base FF's own files are untouched, just copied alongside
    assert (ext_dir / "aminoacids.rtp").exists()


def test_build_ncaa_forcefield_includes_params_itp(tmp_path):
    base_ff = _make_fake_base_ff(tmp_path)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "residue.rtp").write_text("[ XAA ]\n  [ atoms ]\n")
    (bundle / "residue.hdb").write_text("XAA        0\n")
    (bundle / "params.itp").write_text("[ atomtypes ]\n; novel atom type\n")

    ff = CHARMM36mFF()
    output_dir = tmp_path / "stage2"
    output_dir.mkdir()
    ff_flag = ff.build_ncaa_forcefield([bundle], base_ff, output_dir)

    ext_dir = output_dir / f"{ff_flag}.ff"
    included_name = f"{bundle.name}_params.itp"
    assert (ext_dir / included_name).exists()
    assert f'#include "{included_name}"' in (ext_dir / "forcefield.itp").read_text()
