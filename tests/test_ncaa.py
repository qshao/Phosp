from pathlib import Path
import numpy as np
import pytest
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Residue
from Bio.PDB.Atom import Atom

from phosp.modification.ncaa import NcaaModifier, kabsch_fit
from phosp.modification.ncaa_bundle import lint_bundle, parse_rtp_block, parse_hdb_block
from phosp.config import NcaaSite

FIXTURES = Path(__file__).parent / "fixtures"

# A minimal synthetic residue: ALA-like backbone (N/CA/HN/HA/C/O) plus a
# 2-hydrogen CB and a novel 3-hydrogen CX branch, chosen so charges sum to
# exactly 0 and the hdb exercises both single- and multi-hydrogen rules.
_XAA_RTP = """\
[ XAA ]
  [ atoms ]
        N      NH1 -0.5000   1
       HN        H  0.3000   1
       CA      CT1  0.1000   1
       HA      HB1  0.1000   1
       CB      CT2 -0.2000   2
      HB1      HA2  0.1000   2
      HB2      HA2  0.1000   2
       CX      CT3 -0.3000   2
     HCX1      HA3  0.1000   2
     HCX2      HA3  0.1000   2
     HCX3      HA3  0.1000   2
        C        C  0.5000   3
        O        O -0.5000   3
  [ bonds ]
       CB    CA
       CX    CB
        N    HN
        N    CA
        C    CA
        C    +N
       CA    HA
       CB   HB1
       CB   HB2
      CX   HCX1
      CX   HCX2
      CX   HCX3
        O     C
  [ impropers ]
        N    -C    CA    HN
        C    CA    +N     O
  [ cmap ]
       -C     N    CA     C    +N
"""

_XAA_HDB = """\
XAA        4
1       1       HN      N       CA      -C
1       5       HA      CA      C       CB      N
2       4       HB      CB      CA      CX
3       4       HCX     CX      CB      CA
"""


def _write_bundle(tmp_path, rtp=_XAA_RTP, hdb=_XAA_HDB, with_template=True) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "residue.rtp").write_text(rtp)
    (bundle / "residue.hdb").write_text(hdb)
    if with_template:
        _write_template_pdb(bundle / "template.pdb")
    return bundle


def _write_template_pdb(path: Path) -> None:
    """Build a small XAA residue with Bio.PDB objects and let PDBIO handle
    fixed-width PDB formatting, rather than hand-authoring columns."""
    structure = Structure.Structure("xaa_template")
    model = Model.Model(0)
    chain = Chain.Chain("A")
    residue = Residue.Residue((" ", 1, " "), "XAA", " ")

    coords = {
        "N": (0.0, 0.0, 0.0), "HN": (0.3, 0.7, 0.2),
        "CA": (1.458, 0.0, 0.0), "HA": (1.85, 0.85, -0.55),
        "CB": (1.98, 1.35, 0.52), "HB1": (1.70, 2.20, 0.0), "HB2": (1.60, 1.40, 1.54),
        "CX": (3.48, 1.32, 0.48), "HCX1": (3.85, 0.40, 0.95),
        "HCX2": (3.85, 2.18, 1.05), "HCX3": (3.85, 1.35, -0.55),
        "C": (1.98, -1.20, -0.76), "O": (1.50, -2.28, -0.50),
    }
    for name, xyz in coords.items():
        element = "H" if name.startswith("H") else name[0]
        residue.add(Atom(name, np.array(xyz), 0.0, 1.0, " ", name, 0, element))

    chain.add(residue)
    model.add(chain)
    structure.add(model)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(path))


def _load_ubiquitin():
    return PDBParser(QUIET=True).get_structure("p", str(FIXTURES / "ubiquitin.pdb"))


# --- kabsch_fit ---------------------------------------------------------

def test_kabsch_fit_recovers_known_rotation_and_translation():
    rng_points = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0],
    ])
    theta = np.pi / 3
    true_R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    true_t = np.array([2.0, -1.0, 0.5])
    target = rng_points @ true_R.T + true_t

    R, t = kabsch_fit(rng_points, target)
    recovered = rng_points @ R.T + t
    assert np.allclose(recovered, target, atol=1e-8)


def test_kabsch_fit_identity_for_already_aligned_points():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    R, t = kabsch_fit(points, points)
    assert np.allclose(R, np.eye(3), atol=1e-8)
    assert np.allclose(t, np.zeros(3), atol=1e-8)


# --- NcaaModifier --------------------------------------------------------

def test_ncaa_modifier_grafts_novel_atom_and_renames_residue(tmp_path):
    bundle = _write_bundle(tmp_path)
    struct = _load_ubiquitin()

    modifier = NcaaModifier(bundle, new_resname="XAA")
    modified = modifier.apply(struct, chain_id="A", resid=48)

    residue = modified[0]["A"][(" ", 48, " ")]
    assert residue.resname == "XAA"
    assert "CX" in residue
    assert "HCX1" in residue and "HCX2" in residue and "HCX3" in residue


def test_ncaa_modifier_replaces_old_side_chain_despite_name_collisions(tmp_path):
    """Regression test: side-chain atom names (CB, CG, ...) follow the same
    Greek-letter convention across every amino acid, so LYS's own CB is NOT
    the template's CB just because they share a name — the old side chain
    (CG/CD/CE/NZ, none of which the XAA template defines) must be fully
    removed, not left dangling alongside the new one."""
    bundle = _write_bundle(tmp_path)
    struct = _load_ubiquitin()
    original_cb = struct[0]["A"][(" ", 48, " ")]["CB"].get_vector().get_array().copy()

    modifier = NcaaModifier(bundle, new_resname="XAA")
    modified = modifier.apply(struct, chain_id="A", resid=48)

    residue = modified[0]["A"][(" ", 48, " ")]
    for stale_name in ("CG", "CD", "CE", "NZ"):
        assert stale_name not in residue, f"{stale_name} from the old LYS side chain should be removed"
    # CB is redefined by the template, not the old residue's CB kept in place
    new_cb = residue["CB"].get_vector().get_array()
    assert not np.allclose(original_cb, new_cb)


def test_ncaa_modifier_preserves_existing_backbone_positions(tmp_path):
    bundle = _write_bundle(tmp_path)
    struct = _load_ubiquitin()
    original_ca = struct[0]["A"][(" ", 48, " ")]["CA"].get_vector().get_array().copy()

    modifier = NcaaModifier(bundle, new_resname="XAA")
    modified = modifier.apply(struct, chain_id="A", resid=48)

    new_ca = modified[0]["A"][(" ", 48, " ")]["CA"].get_vector().get_array()
    assert np.allclose(original_ca, new_ca)


# --- config.NcaaSite ------------------------------------------------------

def test_ncaasite_rejects_reserved_resname(tmp_path):
    with pytest.raises(Exception, match="collides"):
        NcaaSite(chain="A", resid=1, resname="MET", new_resname="ALA", bundle_dir=tmp_path)


def test_ncaasite_requires_existing_bundle_dir():
    with pytest.raises(Exception, match="bundle_dir not found"):
        NcaaSite(chain="A", resid=1, resname="MET", new_resname="XAA", bundle_dir=Path("/nonexistent/bundle/dir"))


def test_ncaasite_accepts_valid_bundle(tmp_path):
    site = NcaaSite(chain="A", resid=1, resname="MET", new_resname="XAA", bundle_dir=tmp_path)
    assert site.new_resname == "XAA"


# --- ncaa_bundle parsing / linting ----------------------------------------

def test_parse_rtp_block_extracts_sections():
    parsed = parse_rtp_block(_XAA_RTP)
    assert parsed["resname"] == "XAA"
    assert {a["name"] for a in parsed["atoms"]} == {
        "N", "HN", "CA", "HA", "CB", "HB1", "HB2", "CX", "HCX1", "HCX2", "HCX3", "C", "O",
    }
    assert ("C", "+N") in parsed["bonds"]


def test_parse_hdb_block_extracts_rules():
    parsed = parse_hdb_block(_XAA_HDB)
    assert parsed["resname"] == "XAA"
    assert len(parsed["rules"]) == 4


def test_lint_bundle_valid_passes(tmp_path):
    bundle = _write_bundle(tmp_path)
    assert lint_bundle(bundle) == []


def test_lint_bundle_missing_files(tmp_path):
    bundle = tmp_path / "empty_bundle"
    bundle.mkdir()
    errors = lint_bundle(bundle)
    assert any("missing required file" in e for e in errors)


def test_lint_bundle_malformed_bonds_line_does_not_crash(tmp_path):
    """Regression test: a truncated [ bonds ] line (e.g. one stray token) must
    surface as a clean lint error, not an unhandled ValueError from unpacking
    a too-short tuple downstream in lint_bundle."""
    bad_rtp = _XAA_RTP.replace("       CX    CB", "       CX")
    bundle = _write_bundle(tmp_path, rtp=bad_rtp)
    errors = lint_bundle(bundle)
    assert any("malformed [ bonds ] line" in e for e in errors)


def test_lint_bundle_bad_atom_ref_in_bonds(tmp_path):
    bad_rtp = _XAA_RTP.replace("       CX    CB", "       CX    NOTREAL")
    bundle = _write_bundle(tmp_path, rtp=bad_rtp)
    errors = lint_bundle(bundle)
    assert any("NOTREAL" in e for e in errors)


def test_lint_bundle_non_integer_charge(tmp_path):
    bad_rtp = _XAA_RTP.replace("       CX      CT3 -0.3000   2", "       CX      CT3 -0.3500   2")
    bundle = _write_bundle(tmp_path, rtp=bad_rtp)
    errors = lint_bundle(bundle)
    assert any("sum to" in e for e in errors)


def test_lint_bundle_hdb_hydrogen_not_declared(tmp_path):
    bad_hdb = _XAA_HDB.replace("HB      CB      CA      CX", "HQ      CB      CA      CX")
    bundle = _write_bundle(tmp_path, hdb=bad_hdb)
    errors = lint_bundle(bundle)
    assert any("HQ" in e for e in errors)
