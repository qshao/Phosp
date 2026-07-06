from pathlib import Path
import MDAnalysis as mda
import pytest
from phosp.plugins.analysis.base import PROTEIN_SELECTION
from phosp.plugins.analysis.radius_of_gyration import RadiusOfGyrationPlugin
from phosp.plugins.analysis.sasa import SASAPlugin

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def universe_with_tpo(tmp_path):
    """ubiquitin.pdb with THR66 renamed to TPO66 — a bare 'protein' selection
    macro (confirmed against MDAnalysis's ProteinSelection) does not recognize
    TPO, silently excluding a real phosphosite from any plugin using it."""
    text = (FIXTURES / "ubiquitin.pdb").read_text()
    patched = text.replace("THR A  66", "TPO A  66")
    assert patched != text, "fixture no longer contains the expected THR66 line"
    pdb_path = tmp_path / "ubiquitin_tpo.pdb"
    pdb_path.write_text(patched)
    return mda.Universe(str(pdb_path))


def test_protein_selection_includes_phospho_residue(universe_with_tpo):
    bare = universe_with_tpo.select_atoms("protein")
    extended = universe_with_tpo.select_atoms(PROTEIN_SELECTION)
    assert 66 not in set(bare.resids), "bare 'protein' macro should still exclude TPO"
    assert 66 in set(extended.resids), "PROTEIN_SELECTION must include TPO"


def test_radius_of_gyration_includes_phospho_residue(universe_with_tpo):
    plugin = RadiusOfGyrationPlugin()
    df = plugin.run(universe_with_tpo, {})
    protein_all = universe_with_tpo.select_atoms(PROTEIN_SELECTION)
    protein_without_tpo = universe_with_tpo.select_atoms("protein")
    assert protein_all.radius_of_gyration() != protein_without_tpo.radius_of_gyration()
    assert df["rg_angstrom"].iloc[0] == pytest.approx(protein_all.radius_of_gyration())


def test_sasa_target_selection_includes_phospho_residue(universe_with_tpo):
    plugin = SASAPlugin()
    df = plugin.run(universe_with_tpo, {"residues": [66]})
    assert (df["sasa_target_angstrom2"] > 0).all()
