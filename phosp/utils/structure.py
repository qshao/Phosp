from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from urllib.request import urlopen, urlretrieve

from Bio.PDB import PDBIO, PDBParser, Select

logger = logging.getLogger(__name__)


class _CleanSelect(Select):
    def __init__(self, keep_hetatm: list[str]) -> None:
        self._keep = set(keep_hetatm)

    def accept_residue(self, residue):
        hetflag = residue.get_id()[0]
        if hetflag == " ":
            return True
        if hetflag == "W":
            return False
        return residue.get_resname().strip() in self._keep


def fetch_structure(
    source: str,
    path: Path | None,
    uniprot_id: str | None,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "input.pdb"
    if source == "pdb":
        shutil.copy(path, dest)
        return dest
    return _fetch_uniprot(uniprot_id, dest)


def _fetch_uniprot(uniprot_id: str, dest: Path) -> Path:
    af_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    try:
        logger.info("Fetching AlphaFold structure for %s", uniprot_id)
        urlretrieve(af_url, dest)
        return dest
    except Exception:
        logger.warning("AlphaFold fetch failed; trying RCSB for %s", uniprot_id)
        return _fetch_rcsb_by_uniprot(uniprot_id, dest)


def _fetch_rcsb_by_uniprot(uniprot_id: str, dest: Path) -> Path:
    query_url = (
        "https://search.rcsb.org/rcsbsearch/v2/query?json="
        '{"query":{"type":"terminal","service":"text",'
        '"parameters":{"attribute":"rcsb_polymer_entity_container_identifiers'
        '.reference_sequence_identifiers.database_accession",'
        f'"operator":"exact_match","value":"{uniprot_id}"'
        '}},"return_type":"entry","request_options":{"results_verbosity":"minimal","paginate":{"start":0,"rows":1}}}'
    )
    with urlopen(query_url) as resp:
        data = json.loads(resp.read())
    hits = data.get("result_set", [])
    if not hits:
        raise RuntimeError(f"No PDB structure found for UniProt {uniprot_id}")
    pdb_id = hits[0]["identifier"]
    pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    logger.info("Downloading %s from RCSB", pdb_id)
    urlretrieve(pdb_url, dest)
    return dest


def clean_structure(
    pdb: Path, output: Path, keep_hetatm: list[str] | None = None
) -> Path:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output), _CleanSelect(keep_hetatm or []))
    return output


def protonate_structure(
    pdb: Path, output: Path, ph: float = 7.4, pdb2pqr_binary: str = "pdb2pqr",
    timeout: int | None = None,
) -> Path:
    pqr_output = output.with_suffix(".pqr")
    cmd = [
        pdb2pqr_binary,
        "--ff=CHARMM",
        "--titration-state-method=propka",
        f"--with-ph={ph}",
        "--pdb-output", str(output),
        str(pdb),
        str(pqr_output),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        minutes = timeout // 60 if timeout else "?"
        raise RuntimeError(f"PDB2PQR timed out after {minutes} minutes")
    if result.returncode != 0:
        raise RuntimeError(f"PDB2PQR failed:\n{result.stderr[-2000:]}")
    return output
