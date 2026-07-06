from pathlib import Path
import MDAnalysis as mda
import pytest
from phosp.plugins.analysis.rmsd import RMSDPlugin
from phosp.plugins.analysis.rmsf import RMSFPlugin
from phosp.plugins.analysis.radius_of_gyration import RadiusOfGyrationPlugin

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def universe():
    return mda.Universe(
        str(FIXTURES / "mini_traj.pdb"),
        str(FIXTURES / "mini_traj.xtc"),
    )


def test_rmsd_plugin_returns_dataframe(universe):
    plugin = RMSDPlugin()
    df = plugin.run(universe, {"selection": "backbone"})
    assert "rmsd_angstrom" in df.columns
    assert len(df) == len(universe.trajectory)


def test_rmsd_plugin_defaults_to_ca_only(universe):
    """Project default is CA-only RMSD, not the full backbone (N, CA, C, O)."""
    import numpy as np
    from unittest.mock import patch
    plugin = RMSDPlugin()
    with patch("phosp.plugins.analysis.rmsd.rms.RMSD") as mock_rmsd:
        mock_rmsd.return_value.run.return_value = mock_rmsd.return_value
        mock_rmsd.return_value.results.rmsd = np.zeros((len(universe.trajectory), 3))
        plugin.run(universe, {})
    assert mock_rmsd.call_args.kwargs["select"] == "name CA"


def test_rmsf_plugin_returns_per_residue(universe):
    plugin = RMSFPlugin()
    df = plugin.run(universe, {"selection": "name CA"})
    assert "rmsf_angstrom" in df.columns
    assert "resid" in df.columns


def test_rmsf_plugin_does_not_mutate_shared_universe(universe):
    """stage4_analyze.py passes one Universe to every plugin in sequence —
    RMSF's alignment must run on its own copy, not the shared one."""
    import numpy as np
    universe.trajectory[0]
    before = universe.atoms.positions.copy()

    RMSFPlugin().run(universe, {"selection": "name CA"})

    universe.trajectory[0]
    after = universe.atoms.positions
    assert np.allclose(before, after)


def test_rmsf_plugin_aligns_before_computing(universe):
    """Regression test: unaligned RMSF mixes whole-body rotation/translation
    into per-residue fluctuation, inflating it well above the aligned value."""
    from MDAnalysis.analysis import rms

    plugin = RMSFPlugin()
    aligned_df = plugin.run(universe, {"selection": "name CA"})

    ca = universe.select_atoms("name CA")
    naive_rmsf = rms.RMSF(ca).run().results.rmsf

    assert aligned_df["rmsf_angstrom"].mean() < naive_rmsf.mean()


def test_rg_plugin_returns_per_frame(universe):
    plugin = RadiusOfGyrationPlugin()
    df = plugin.run(universe, {})
    assert "rg_angstrom" in df.columns
    assert len(df) == len(universe.trajectory)


def test_rmsd_plugin_plot_returns_figure(universe):
    import matplotlib
    matplotlib.use("Agg")
    plugin = RMSDPlugin()
    df = plugin.run(universe, {})
    fig = plugin.plot(df)
    assert fig is not None
