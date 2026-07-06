from pathlib import Path
from unittest.mock import patch, MagicMock
import MDAnalysis as mda
import pytest
from phosp.plugins.analysis.pca import PCAPlugin
from phosp.plugins.analysis.dccm import DCCMPlugin
from phosp.plugins.analysis.mmpbsa import MMPBSAPlugin

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def universe():
    return mda.Universe(
        str(FIXTURES / "mini_traj.pdb"),
        str(FIXTURES / "mini_traj.xtc"),
    )


def test_pca_returns_two_components(universe):
    plugin = PCAPlugin()
    df = plugin.run(universe, {"selection": "name CA"})
    assert "pc1" in df.columns
    assert "pc2" in df.columns
    assert len(df) == len(universe.trajectory)


def test_pca_does_not_mutate_shared_universe(universe):
    """stage4_analyze.py passes one Universe to every plugin in sequence —
    PCA's align=True must run on its own copy, not the shared one."""
    import numpy as np
    universe.trajectory[0]
    before = universe.atoms.positions.copy()

    PCAPlugin().run(universe, {"selection": "name CA"})

    universe.trajectory[0]
    after = universe.atoms.positions
    assert np.allclose(before, after)


def test_dccm_returns_square_matrix(universe):
    plugin = DCCMPlugin()
    df = plugin.run(universe, {"selection": "name CA"})
    assert "dcc" in df.columns
    ca = universe.select_atoms("name CA")
    n = len(ca.residues)
    assert len(df) == n * n


def test_mmpbsa_raises_if_not_installed(tmp_path):
    with patch("shutil.which", return_value=None):
        plugin = MMPBSAPlugin()
        fake_u = MagicMock()
        with pytest.raises(RuntimeError, match="gmx_MMPBSA not found"):
            plugin.run(fake_u, {})
