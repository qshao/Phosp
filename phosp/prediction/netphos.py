from __future__ import annotations
import logging
import re
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_TYPE_MAP = {"S": "pSer", "T": "pThr", "Y": "pTyr"}
_RESNAME_MAP = {"S": "SER", "T": "THR", "Y": "TYR"}


def _parse_netphos_output(output: str, threshold: float = 0.5) -> list[dict]:
    results = []
    for line in output.splitlines():
        # Example: "ubiquitin   42 LIFAGKQLEDGR   0.801 +   S"
        m = re.match(
            r"\S+\s+(\d+)\s+\S+\s+([\d.]+)\s+(\+?)\s+([STY])", line
        )
        if not m:
            continue
        resid, score_str, plus, aa = m.group(1, 2, 3, 4)
        score = float(score_str)
        if score < threshold:
            continue
        results.append({
            "resid": int(resid),
            "resname": _RESNAME_MAP[aa],
            "phospho_type": _TYPE_MAP[aa],
            "score": score,
            "chain": "A",  # NetPhos does not report chain; default A
        })
    return results


class NetPhos:
    def predict(self, pdb: Path, threshold: float = 0.5) -> list[dict]:
        exe = shutil.which("netphos") or shutil.which("netphos3.1")
        if not exe:
            raise RuntimeError(
                "NetPhos not found in PATH. Install NetPhos 3.1 or set PATH correctly."
            )
        result = subprocess.run([exe, str(pdb)], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"NetPhos failed:\n{result.stderr[-1000:]}")
        return _parse_netphos_output(result.stdout, threshold=threshold)
