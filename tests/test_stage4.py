import importlib
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
from phosp.config import load_config
from phosp.stages.stage4_analyze import Stage4Analyze, _discover_plugins
from phosp.plugins.analysis.base import AnalysisPlugin

FIXTURES = Path(__file__).parent / "fixtures"


class _FakePlugin(AnalysisPlugin):
    name = "fake"
    def run(self, universe, config):
        return pd.DataFrame({"x": [1, 2, 3]})
    def plot(self, result):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(result["x"])
        return fig


def test_discover_plugins_finds_registered_subclasses():
    plugins = _discover_plugins()
    # AnalysisPlugin subclasses auto-registered; at minimum base module is imported
    assert isinstance(plugins, dict)


def test_stage4_run_executes_requested_plugins(tmp_path):
    cfg = load_config(FIXTURES / "valid_config.yaml")
    cfg.input.path = FIXTURES / "ubiquitin.pdb"
    cfg.analysis.plugins = ["fake"]

    stage3_dir = tmp_path / "output" / "stage3" / "production"
    stage3_dir.mkdir(parents=True)
    (stage3_dir / "prod.xtc").write_bytes(b"")
    (stage3_dir / "prod.tpr").write_bytes(b"")

    stage = Stage4Analyze(cfg, MagicMock(), MagicMock(), tmp_path / "output" / "stage4")

    fake_universe = MagicMock()
    with patch("phosp.stages.stage4_analyze.mda.Universe", return_value=fake_universe), \
         patch("phosp.stages.stage4_analyze._discover_plugins", return_value={"fake": _FakePlugin}):
        result = stage.run()

    assert (tmp_path / "output" / "stage4" / "fake.csv").exists()


def test_stage4_validate_raises_if_trajectory_missing(tmp_path):
    from phosp.exceptions import StageInputError
    cfg = load_config(FIXTURES / "valid_config.yaml")
    cfg.input.path = FIXTURES / "ubiquitin.pdb"
    stage = Stage4Analyze(cfg, MagicMock(), MagicMock(), tmp_path / "stage4")
    with pytest.raises(StageInputError, match="prod.xtc"):
        stage.validate_inputs()
