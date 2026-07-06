from pathlib import Path
import MDAnalysis as mda
import pytest
from phosp.plugins.analysis.sasa import SASAPlugin

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def universe():
    return mda.Universe(
        str(FIXTURES / "mini_traj.pdb"),
        str(FIXTURES / "mini_traj.xtc"),
    )


def test_sasa_whole_protein(universe):
    plugin = SASAPlugin()
    df = plugin.run(universe, {"residues": []})
    assert "sasa_angstrom2" in df.columns
    assert len(df) == len(universe.trajectory)
    assert (df["sasa_angstrom2"] > 0).all()


def test_sasa_specific_residue(universe):
    resids = list(universe.select_atoms("protein").residues.resids[:2])
    plugin = SASAPlugin()
    df = plugin.run(universe, {"residues": resids})
    assert len(df) == len(universe.trajectory)
    assert "sasa_target_angstrom2" in df.columns
    assert "sasa_angstrom2" in df.columns
    assert (df["sasa_target_angstrom2"] > 0).all()
    # full-protein total must stay far larger than any single-residue slice
    assert (df["sasa_target_angstrom2"] < df["sasa_angstrom2"]).all()


def test_sasa_target_computed_in_full_protein_context(universe):
    """Regression test: per-residue SASA must reflect steric shielding from
    the rest of the protein, not the SASA of the residue in isolation."""
    import freesasa
    mid_chain_resid = int(universe.select_atoms("protein").residues.resids[100])

    plugin = SASAPlugin()
    df = plugin.run(universe, {"residues": [mid_chain_resid]})
    in_context = df["sasa_target_angstrom2"].iloc[0]

    # Reproduce the old (buggy) isolated-fragment calculation directly.
    isolated_atoms = universe.select_atoms(f"protein and resid {mid_chain_resid}")
    universe.trajectory[0]
    classifier = freesasa.Classifier()
    radii = []
    for atom in isolated_atoms:
        try:
            r = classifier.radius(atom.resname, atom.name)
        except Exception:
            r = -1.0
        radii.append(r if r >= 0 else 1.5)
    isolated_result = freesasa.calcCoord(isolated_atoms.positions.flatten().tolist(), radii)
    isolated_area = isolated_result.totalArea()

    # Shielding from neighboring residues can only reduce (or leave equal)
    # the exposed area relative to the isolated fragment.
    assert in_context < isolated_area
