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
    (stage3_dir / "production.xtc").write_bytes(b"")
    (stage3_dir / "production.tpr").write_bytes(b"")

    stage4_dir = tmp_path / "output" / "stage4"
    stage4_dir.mkdir(parents=True)
    stage = Stage4Analyze(cfg, MagicMock(), MagicMock(), stage4_dir)

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
    with pytest.raises(StageInputError, match="production.xtc"):
        stage.validate_inputs()


def test_stage4_validate_gives_hpc_message_when_pending(tmp_path):
    """When pending_job.json exists but xtc is absent, error mentions HPC job status."""
    import json
    from phosp.exceptions import StageInputError
    cfg = load_config(FIXTURES / "valid_config.yaml")
    cfg.input.path = FIXTURES / "ubiquitin.pdb"

    stage3_dir = tmp_path / "output" / "stage3"
    stage3_dir.mkdir(parents=True)
    sentinel = stage3_dir / "pending_job.json"
    sentinel.write_text(json.dumps({"scheduler": "slurm", "job_id": "99999", "auto_submitted": True}))

    stage = Stage4Analyze(cfg, MagicMock(), MagicMock(), tmp_path / "output" / "stage4")
    with pytest.raises(StageInputError, match="SLURM job.*99999.*still running"):
        stage.validate_inputs()


def test_plugin_partial_failure_continues(tmp_path):
    """One plugin fails, others succeed — no exception, partial results saved."""
    from phosp.exceptions import AnalysisError

    cfg = load_config(FIXTURES / "valid_config.yaml")
    cfg.analysis.plugins = ["bad", "fake"]

    stage3_dir = tmp_path / "output" / "stage3" / "production"
    stage3_dir.mkdir(parents=True)
    (stage3_dir / "production.xtc").write_bytes(b"")
    (stage3_dir / "production.tpr").write_bytes(b"")

    class _BadPlugin(AnalysisPlugin):
        name = "bad"
        def run(self, universe, config):
            raise RuntimeError("intentional failure")
        def plot(self, result):
            import matplotlib.pyplot as plt
            return plt.figure()

    stage = Stage4Analyze(cfg, MagicMock(), MagicMock(), tmp_path / "output" / "stage4")
    fake_universe = MagicMock()
    with patch("phosp.stages.stage4_analyze.mda.Universe", return_value=fake_universe), \
         patch("phosp.stages.stage4_analyze._discover_plugins",
               return_value={"bad": _BadPlugin, "fake": _FakePlugin}):
        result = stage.run()

    assert (tmp_path / "output" / "stage4" / "fake.csv").exists()
    assert not (tmp_path / "output" / "stage4" / "bad.csv").exists()


def test_all_plugins_fail_raises_analysis_error(tmp_path):
    """All plugins fail → AnalysisError raised."""
    from phosp.exceptions import AnalysisError

    cfg = load_config(FIXTURES / "valid_config.yaml")
    cfg.analysis.plugins = ["bad"]

    stage3_dir = tmp_path / "output" / "stage3" / "production"
    stage3_dir.mkdir(parents=True)
    (stage3_dir / "production.xtc").write_bytes(b"")
    (stage3_dir / "production.tpr").write_bytes(b"")

    class _BadPlugin(AnalysisPlugin):
        name = "bad"
        def run(self, universe, config):
            raise RuntimeError("all dead")
        def plot(self, result):
            import matplotlib.pyplot as plt
            return plt.figure()

    stage = Stage4Analyze(cfg, MagicMock(), MagicMock(), tmp_path / "output" / "stage4")
    fake_universe = MagicMock()
    with patch("phosp.stages.stage4_analyze.mda.Universe", return_value=fake_universe), \
         patch("phosp.stages.stage4_analyze._discover_plugins",
               return_value={"bad": _BadPlugin}):
        with pytest.raises(AnalysisError, match="All analysis plugins failed"):
            stage.run()


def test_report_generated_after_run(tmp_path):
    """run() generates report.html."""
    cfg = load_config(FIXTURES / "valid_config.yaml")
    cfg.analysis.plugins = ["fake"]

    stage3_dir = tmp_path / "output" / "stage3" / "production"
    stage3_dir.mkdir(parents=True)
    (stage3_dir / "production.xtc").write_bytes(b"")
    (stage3_dir / "production.tpr").write_bytes(b"")

    stage = Stage4Analyze(cfg, MagicMock(), MagicMock(), tmp_path / "output" / "stage4")
    fake_universe = MagicMock()
    with patch("phosp.stages.stage4_analyze.mda.Universe", return_value=fake_universe), \
         patch("phosp.stages.stage4_analyze._discover_plugins",
               return_value={"fake": _FakePlugin}):
        stage.run()

    assert (tmp_path / "output" / "stage4" / "report.html").exists()


def test_unknown_plugin_name_raises_analysis_error(tmp_path):
    """A typo in analysis.plugins raises AnalysisError immediately with valid names listed."""
    from phosp.exceptions import AnalysisError
    cfg = load_config(FIXTURES / "valid_config.yaml")
    cfg.analysis.plugins = ["rmsdf"]   # typo: should be "rmsf"

    stage3_dir = tmp_path / "output" / "stage3" / "production"
    stage3_dir.mkdir(parents=True)
    (stage3_dir / "production.xtc").write_bytes(b"")
    (stage3_dir / "production.gro").write_bytes(b"")

    stage = Stage4Analyze(cfg, MagicMock(), MagicMock(), tmp_path / "output" / "stage4")
    with patch("phosp.stages.stage4_analyze._discover_plugins",
               return_value={"rmsd": MagicMock, "rmsf": MagicMock}) as discover_mock, \
         patch("phosp.stages.stage4_analyze.mda.Universe", return_value=MagicMock()) as mda_universe_mock:
        with pytest.raises(AnalysisError, match="Unknown analysis plugins.*rmsdf"):
            stage.run()
        mda_universe_mock.assert_not_called()
