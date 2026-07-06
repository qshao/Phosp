from pathlib import Path
import pytest
from phosp.forcefields.charmm36m import CHARMM36mFF
from phosp.forcefields.amber_ff14sb import AMBERff14SBFF
from phosp.forcefields.discovery import discover_top_dir


def test_discover_top_dir_prefers_gmxlib_even_when_ff_not_installed_there_yet(tmp_path, monkeypatch):
    """Regression test: GMXLIB (when set) is what real `gmx -ff <name>` actually
    searches first, regardless of whether the force field is installed there
    yet — discover_top_dir must match that, not silently fall through to the
    Data-prefix branch just because GMXLIB's directory is currently empty."""
    monkeypatch.setenv("GMXLIB", str(tmp_path))
    result = discover_top_dir("gmx", "charmm36m-jul2022.ff")
    assert result == tmp_path


def test_discover_top_dir_prefers_gmxlib_when_ff_present(tmp_path, monkeypatch):
    (tmp_path / "charmm36m-jul2022.ff").mkdir()
    monkeypatch.setenv("GMXLIB", str(tmp_path))
    result = discover_top_dir("gmx", "charmm36m-jul2022.ff")
    assert result == tmp_path


def test_discover_top_dir_falls_back_to_data_prefix_when_gmxlib_unset(monkeypatch):
    monkeypatch.delenv("GMXLIB", raising=False)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: type("R", (), {"stdout": "Data prefix:  /usr\n"})(),
    )
    result = discover_top_dir("gmx", "charmm36m-jul2022.ff")
    assert result == Path("/usr/share/gromacs/top")


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
    included_name = "bundle0_params.itp"
    assert (ext_dir / included_name).exists()
    assert f'#include "{included_name}"' in (ext_dir / "forcefield.itp").read_text()


def test_build_ncaa_forcefield_dedups_repeated_bundle_dir(tmp_path):
    """Regression test: the same ncAA bundle applied at two different sites
    (a legitimate config: [site1, site2] both referencing the same bundle_dir)
    must be merged into ncaa.rtp/ncaa.hdb only once, not duplicated."""
    base_ff = _make_fake_base_ff(tmp_path)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "residue.rtp").write_text("[ XAA ]\n  [ atoms ]\n")
    (bundle / "residue.hdb").write_text("XAA        0\n")

    ff = CHARMM36mFF()
    output_dir = tmp_path / "stage2"
    output_dir.mkdir()
    ff_flag = ff.build_ncaa_forcefield([bundle, bundle], base_ff, output_dir)

    ext_dir = output_dir / f"{ff_flag}.ff"
    assert (ext_dir / "ncaa.rtp").read_text().count("[ XAA ]") == 1


def test_build_ncaa_forcefield_handles_same_basename_different_bundles(tmp_path):
    """Regression test: two different bundle directories that happen to share
    a basename (e.g. "expA/bundle", "expB/bundle") must not collide — each
    bundle's params.itp must survive and be included independently."""
    base_ff = _make_fake_base_ff(tmp_path)
    exp_a = tmp_path / "expA" / "bundle"
    exp_b = tmp_path / "expB" / "bundle"
    for bundle, resname, marker in [(exp_a, "AAA", "marker-A"), (exp_b, "BBB", "marker-B")]:
        bundle.mkdir(parents=True)
        (bundle / "residue.rtp").write_text(f"[ {resname} ]\n  [ atoms ]\n")
        (bundle / "residue.hdb").write_text(f"{resname}        0\n")
        (bundle / "params.itp").write_text(f"; {marker}\n")

    ff = CHARMM36mFF()
    output_dir = tmp_path / "stage2"
    output_dir.mkdir()
    ff_flag = ff.build_ncaa_forcefield([exp_a, exp_b], base_ff, output_dir)
    ext_dir = output_dir / f"{ff_flag}.ff"

    itp_files = sorted(ext_dir.glob("bundle*_params.itp"))
    assert len(itp_files) == 2
    contents = {p.read_text() for p in itp_files}
    assert contents == {"; marker-A\n", "; marker-B\n"}
    forcefield_itp = (ext_dir / "forcefield.itp").read_text()
    assert forcefield_itp.count("#include") == 4  # 2 base (ffnonbonded/ffbonded) + 2 bundle
