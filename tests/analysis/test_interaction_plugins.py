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


def test_salt_bridges_includes_phosphate_oxygens(tmp_path):
    """Regression test: the phosphate group (SEP/TPO/PTR) is a strongly
    anionic side chain that should count as 'acidic' alongside ASP/GLU —
    it was previously excluded entirely."""
    from Bio.PDB import PDBParser, PDBIO
    from phosp.modification.pthr import PThrModifier

    struct = PDBParser(QUIET=True).get_structure("p", str(FIXTURES / "ubiquitin.pdb"))
    modified = PThrModifier(forcefield="charmm36m").apply(struct, chain_id="A", resid=66)
    pdb_path = tmp_path / "ubiquitin_tpo.pdb"
    io = PDBIO()
    io.set_structure(modified)
    io.save(str(pdb_path))

    u = mda.Universe(str(pdb_path))
    plugin = SaltBridgesPlugin()
    acidic = u.select_atoms(
        "((resname ASP GLU) and (name OD1 OD2 OE1 OE2)) "
        "or ((resname SEP TPO PTR) and (name O1P O2P O3P))"
    )
    assert "TPO" in acidic.resnames
    assert {"O1P", "O2P", "O3P"} <= set(a.name for a in acidic if a.resname == "TPO")


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


def test_mmpbsa_writes_scratch_files_to_work_dir_not_trajectory_dir(tmp_path):
    """Regression test: mmpbsa's scratch/output files must land in the stage4
    output dir passed via _work_dir, not stage3's already-finalized
    production/ directory (Path(traj).parent) — the latter would be polluted
    on every stage4 re-run."""
    stage3_production = tmp_path / "stage3" / "production"
    stage3_production.mkdir(parents=True)
    stage4_out = tmp_path / "stage4"
    stage4_out.mkdir()

    with patch("phosp.plugins.analysis.mmpbsa.shutil.which", return_value="/usr/bin/gmx_MMPBSA"), \
         patch("phosp.plugins.analysis.mmpbsa.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        MMPBSAPlugin().run(_mock_universe(stage3_production), {"_work_dir": stage4_out})

    assert (stage4_out / "mmpbsa.in").exists()
    assert not (stage3_production / "mmpbsa.in").exists()
    assert mock_run.call_args.kwargs["cwd"] == stage4_out


def test_mmpbsa_raises_clear_error_on_timeout(tmp_path):
    work_dir = tmp_path / "stage4"
    work_dir.mkdir()
    with patch("phosp.plugins.analysis.mmpbsa.shutil.which", return_value="/usr/bin/gmx_MMPBSA"), \
         patch("phosp.plugins.analysis.mmpbsa.subprocess.run",
               side_effect=subprocess.TimeoutExpired(cmd="gmx_MMPBSA", timeout=300)):
        with pytest.raises(RuntimeError, match="gmx_MMPBSA timed out"):
            MMPBSAPlugin().run(_mock_universe(work_dir), {"_timeout_minutes": 5})
