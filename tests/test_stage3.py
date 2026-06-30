from pathlib import Path
from unittest.mock import patch, MagicMock, call
import pytest
from phosp.config import load_config
from phosp.stages.stage3_simulate import Stage3Simulate
from phosp.exceptions import StageInputError
from phosp.engines.base import SimulationResult

FIXTURES = Path(__file__).parent / "fixtures"


def _make_stage3(tmp_path):
    cfg = load_config(FIXTURES / "valid_config.yaml")
    cfg.input.path = FIXTURES / "ubiquitin.pdb"
    stage2_dir = tmp_path / "output" / "stage2"
    stage2_dir.mkdir(parents=True)
    for f in ["topol.top", "ions.gro",
              "minimization.mdp", "nvt.mdp", "npt.mdp", "production.mdp"]:
        (stage2_dir / f).write_text(f"; fake {f}")
    engine = MagicMock()
    engine.run_phase.return_value = SimulationResult(
        phase="any", output_dir=tmp_path, success=True,
        log_path=tmp_path / "fake.log"
    )
    stage3_dir = tmp_path / "output" / "stage3"
    stage3_dir.mkdir(parents=True)
    return Stage3Simulate(cfg, engine, MagicMock(), stage3_dir)


def test_stage3_validate_raises_if_stage2_missing(tmp_path):
    cfg = load_config(FIXTURES / "valid_config.yaml")
    cfg.input.path = FIXTURES / "ubiquitin.pdb"
    stage = Stage3Simulate(cfg, MagicMock(), MagicMock(), tmp_path / "stage3")
    with pytest.raises(StageInputError, match="ions.gro"):
        stage.validate_inputs()


def test_stage3_runs_four_phases(tmp_path):
    stage = _make_stage3(tmp_path)
    result = stage.run()
    assert stage.engine.run_phase.call_count == 4
    phases_called = [c.kwargs["phase"] for c in stage.engine.run_phase.call_args_list]
    assert phases_called == ["minimization", "nvt", "npt", "production"]


def test_stage3_hpc_mode_writes_script_not_runs(tmp_path):
    stage = _make_stage3(tmp_path)
    stage.config.simulation.hpc.enabled = True
    stage.config.simulation.hpc.auto_submit = False
    result = stage.run()
    stage.engine.generate_hpc_script.assert_called_once()
    stage.engine.run_phase.assert_not_called()
