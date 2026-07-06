from __future__ import annotations
from pathlib import Path

# Backbone link tokens valid in an .rtp [ bonds ]/[ impropers ]/[ cmap ] block
# even though they aren't declared in this residue's own [ atoms ] section —
# they refer to the previous/next residue's C/N (see aminoacids.rtp: "C +N").
_BACKBONE_TOKENS = {"-C", "+N", "-CA", "+CA"}


def parse_rtp_block(text: str) -> dict:
    """Parse one [ RESNAME ] ... rtp block into its atoms/bonds/impropers/cmap."""
    lines = [l.split(";", 1)[0].rstrip() for l in text.splitlines()]
    lines = [l for l in lines if l.strip()]
    if not lines or not lines[0].strip().startswith("["):
        raise ValueError("residue.rtp must start with a [ RESNAME ] header")

    resname = lines[0].strip().strip("[]").strip()
    section = None
    atoms, bonds, impropers, cmap = [], [], [], []
    for line in lines[1:]:
        stripped = line.strip()
        if stripped.startswith("["):
            section = stripped.strip("[]").strip()
            continue
        fields = stripped.split()
        if section == "atoms":
            if len(fields) < 4:
                raise ValueError(f"malformed [ atoms ] line: {line!r}")
            atoms.append({"name": fields[0], "type": fields[1],
                           "charge": float(fields[2]), "cgnr": fields[3]})
        elif section == "bonds":
            bonds.append(tuple(fields[:2]))
        elif section == "impropers":
            impropers.append(tuple(fields[:4]))
        elif section == "cmap":
            cmap.append(tuple(fields[:5]))
    return {"resname": resname, "atoms": atoms, "bonds": bonds,
            "impropers": impropers, "cmap": cmap}


def parse_hdb_block(text: str) -> dict:
    """Parse one <RESNAME> <n_rules> ... hdb block into its hydrogen-building rules."""
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    if not lines:
        raise ValueError("residue.hdb is empty")
    header = lines[0].split()
    if len(header) != 2:
        raise ValueError(f"malformed hdb header line: {lines[0]!r}")
    resname, n_rules = header[0], int(header[1])
    rules = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"malformed hdb rule line: {line!r}")
        rules.append({"n_h": int(fields[0]), "method": int(fields[1]),
                       "h_name": fields[2], "refs": fields[3:]})
    if len(rules) != n_rules:
        raise ValueError(
            f"hdb header declares {n_rules} rules but found {len(rules)}"
        )
    return {"resname": resname, "rules": rules}


def lint_bundle(bundle_dir: Path) -> list[str]:
    """Validate an ncAA parameter bundle without invoking GROMACS. Returns a
    list of human-readable error strings; empty means the bundle is internally
    consistent (not a guarantee it's chemically correct)."""
    bundle_dir = Path(bundle_dir)
    errors: list[str] = []

    rtp_path = bundle_dir / "residue.rtp"
    hdb_path = bundle_dir / "residue.hdb"
    template_path = bundle_dir / "template.pdb"
    for required in (rtp_path, hdb_path, template_path):
        if not required.exists():
            errors.append(f"missing required file: {required}")
    if errors:
        return errors

    try:
        rtp = parse_rtp_block(rtp_path.read_text())
    except ValueError as exc:
        return [f"residue.rtp: {exc}"]
    try:
        hdb = parse_hdb_block(hdb_path.read_text())
    except ValueError as exc:
        return [f"residue.hdb: {exc}"]

    atom_names = {a["name"] for a in rtp["atoms"]}

    if hdb["resname"] != rtp["resname"]:
        errors.append(
            f"residue.hdb resname {hdb['resname']!r} != residue.rtp resname {rtp['resname']!r}"
        )

    def _check_atom_ref(ref: str, where: str) -> None:
        if ref in _BACKBONE_TOKENS or ref in atom_names:
            return
        errors.append(f"{where}: atom {ref!r} not declared in [ atoms ] and not a backbone token")

    for a, b in rtp["bonds"]:
        _check_atom_ref(a, "[ bonds ]")
        _check_atom_ref(b, "[ bonds ]")
    for improper in rtp["impropers"]:
        for ref in improper:
            _check_atom_ref(ref, "[ impropers ]")
    for cmap_entry in rtp["cmap"]:
        for ref in cmap_entry:
            _check_atom_ref(ref, "[ cmap ]")

    def _hydrogen_declared(h_name: str) -> bool:
        # A single-hydrogen rule's h_name matches one atom exactly (e.g. "HN").
        # A multi-hydrogen rule's h_name is a stem (e.g. "HB" building HB1/HB2/HB3),
        # which never appears verbatim in [ atoms ] — only the numbered names do.
        if h_name in atom_names:
            return True
        return any(name.startswith(h_name) and name[len(h_name):].isdigit() for name in atom_names)

    for rule in hdb["rules"]:
        if not _hydrogen_declared(rule["h_name"]):
            errors.append(f"residue.hdb: hydrogen {rule['h_name']!r} not declared in residue.rtp [ atoms ]")
        for ref in rule["refs"]:
            _check_atom_ref(ref, "residue.hdb rule refs")

    charge_sum = sum(a["charge"] for a in rtp["atoms"])
    if abs(charge_sum - round(charge_sum)) > 0.01:
        errors.append(f"residue.rtp: atom charges sum to {charge_sum:.4f}, not close to an integer")

    from Bio.PDB import PDBParser
    try:
        template_struct = PDBParser(QUIET=True).get_structure("_ncaa_template", str(template_path))
        template_atom_names = {a.get_name() for a in next(template_struct[0].get_residues()).get_atoms()}
    except (StopIteration, OSError) as exc:
        errors.append(f"template.pdb: could not parse a residue: {exc}")
        template_atom_names = set()

    if template_atom_names and atom_names - template_atom_names:
        errors.append(
            f"template.pdb is missing atoms declared in residue.rtp: "
            f"{sorted(atom_names - template_atom_names)}"
        )

    return errors
