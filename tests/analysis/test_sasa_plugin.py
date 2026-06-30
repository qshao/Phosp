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
