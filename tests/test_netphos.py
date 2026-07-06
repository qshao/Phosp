import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from phosp.prediction.netphos import NetPhos, _parse_netphos_output

SAMPLE_OUTPUT = """\
Name                      Pos Context         S(T/Y)     phos.
ubiquitin                  42 LIFAGKQLEDGR    0.801 +    S
ubiquitin                  66 LEDGRTLSDYNIQ   0.712 +    T
ubiquitin                  59 DQESTLHLVLRL    0.421      S
"""

def test_parse_netphos_output():
    results = _parse_netphos_output(SAMPLE_OUTPUT, threshold=0.5)
    assert len(results) == 2  # 0.801 and 0.712 pass; 0.421 fails
    assert results[0]["resid"] == 42
    assert results[0]["mod_type"] == "pSer"
    assert results[1]["resid"] == 66
    assert results[1]["mod_type"] == "pThr"

def test_parse_filters_by_threshold():
    results = _parse_netphos_output(SAMPLE_OUTPUT, threshold=0.75)
    assert len(results) == 1
    assert results[0]["resid"] == 42

def test_netphos_raises_if_not_installed(tmp_path):
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="NetPhos not found"):
            NetPhos().predict(tmp_path / "fake.pdb")

def test_netphos_passes_timeout_to_subprocess(tmp_path):
    with patch("phosp.prediction.netphos.shutil.which", return_value="/usr/bin/netphos"), \
         patch("phosp.prediction.netphos.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        NetPhos().predict(tmp_path / "fake.pdb", timeout=120)
    assert mock_run.call_args.kwargs["timeout"] == 120

def test_netphos_raises_clear_error_on_timeout(tmp_path):
    with patch("phosp.prediction.netphos.shutil.which", return_value="/usr/bin/netphos"), \
         patch("phosp.prediction.netphos.subprocess.run",
               side_effect=subprocess.TimeoutExpired(cmd="netphos", timeout=120)):
        with pytest.raises(RuntimeError, match="NetPhos timed out"):
            NetPhos().predict(tmp_path / "fake.pdb", timeout=120)
