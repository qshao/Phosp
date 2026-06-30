from pathlib import Path
import MDAnalysis as mda
import pytest
from phosp.plugins.analysis.hbond import HBondPlugin
from phosp.plugins.analysis.salt_bridges import SaltBridgesPlugin
from phosp.plugins.analysis.contacts import ContactsPlugin

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def universe():
    return mda.Universe(
        str(FIXTURES / "mini_traj.pdb"),
        str(FIXTURES / "mini_traj.xtc"),
    )


def test_hbond_plugin_returns_dataframe(universe):
    plugin = HBondPlugin()
    df = plugin.run(universe, {})
    assert "frame" in df.columns


def test_salt_bridges_plugin_returns_dataframe(universe):
    plugin = SaltBridgesPlugin()
    df = plugin.run(universe, {"cutoff_angstrom": 4.0})
    assert "frame" in df.columns


def test_contacts_plugin_shape(universe):
    plugin = ContactsPlugin()
    df = plugin.run(universe, {"cutoff_angstrom": 8.0, "selection": "name CA"})
    assert "frame" in df.columns
    assert "n_contacts" in df.columns
