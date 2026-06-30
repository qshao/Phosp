import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
import MDAnalysis as mda
import pytest
from phosp.plugins.analysis.hbond import HBondPlugin
from phosp.plugins.analysis.salt_bridges import SaltBridgesPlugin
from phosp.plugins.analysis.contacts import ContactsPlugin
from phosp.plugins.analysis.mmpbsa import MMPBSAPlugin

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


def _mock_universe(work_dir: Path) -> MagicMock:
    universe = MagicMock()
    universe.trajectory.filename = str(work_dir / "production.xtc")
    universe.filename = str(work_dir / "production.gro")
    return universe


def test_mmpbsa_passes_timeout_to_subprocess(tmp_path):
    work_dir = tmp_path / "stage4"
    work_dir.mkdir()
    with patch("phosp.plugins.analysis.mmpbsa.shutil.which", return_value="/usr/bin/gmx_MMPBSA"), \
         patch("phosp.plugins.analysis.mmpbsa.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        MMPBSAPlugin().run(_mock_universe(work_dir), {"_timeout_minutes": 5})
    assert mock_run.call_args.kwargs["timeout"] == 300


def test_mmpbsa_raises_clear_error_on_timeout(tmp_path):
    work_dir = tmp_path / "stage4"
    work_dir.mkdir()
    with patch("phosp.plugins.analysis.mmpbsa.shutil.which", return_value="/usr/bin/gmx_MMPBSA"), \
         patch("phosp.plugins.analysis.mmpbsa.subprocess.run",
               side_effect=subprocess.TimeoutExpired(cmd="gmx_MMPBSA", timeout=300)):
        with pytest.raises(RuntimeError, match="gmx_MMPBSA timed out"):
            MMPBSAPlugin().run(_mock_universe(work_dir), {"_timeout_minutes": 5})
