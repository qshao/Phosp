# Worked example: a noncanonical amino acid (ncAA) bundle

This is a complete, runnable example of the ncAA parameter-bundle format
described in the ncAA design notes — homoallylglycine (`HAG`), grafted onto
ubiquitin residue 48.

## The chemistry

Homoallylglycine (2-amino-5-hexenoic acid) has the side chain
`-CH2-CH2-CH=CH2` — a terminal alkene one methylene longer than allylglycine.
It (and its close relatives, e.g. homopropargylglycine) are real ncAAs used
as methionine surrogates for residue-specific incorporation and bio-orthogonal
labeling (the alkene/alkyne handle is used for ring-closing metathesis or
click chemistry after incorporation).

It is **not** present in `charmm36m-jul2022` — but allylglycine (`2AG`, one
carbon shorter) *is*, and every atom type and bonded parameter homoallylglycine
needs (the `CG321`/`CG2D1`/`CG2D2`/`HGA2`/`HGA4`/`HGA5` alkene fragment, and
the bonds/angles/dihedrals connecting them) was confirmed present in the
installed force field's `ffbonded.itp`/`ffnonbonded.itp` before writing
`residue.rtp` — this bundle needs no `params.itp` because nothing here is a
genuinely novel atom type, just a novel *residue* built from existing ones.
`bundle/residue.rtp`'s header comment documents this provenance.

**Why residue 48 (a lysine), not a methionine:** ubiquitin's only methionine
is Met1, the N-terminal residue — and ncAA sites at chain termini aren't
supported yet (would need `.n.tdb`/`.c.tdb` entries too). Grafting onto K48
here is a **structural demonstration of the bundle mechanism only** — it is
not a biologically meaningful substitution. In a real study you'd target an
actual methionine (or another suitable site) via residue-specific incorporation.

## The bundle (`bundle/`)

- `residue.rtp` — the `[ HAG ]` block: atoms (name, force-field type, partial
  charge, charge group), bonds (including the backbone links `-C`/`+N`),
  impropers, and the CHARMM36 backbone CMAP entry.
- `residue.hdb` — hydrogen-building rules, reusing the same construction
  codes CHARMM uses for `2AG`'s own alkene fragment and for a standard `-CH2-`
  (matching `MET`'s own `CB` rule).
- `template.pdb` — one residue's worth of 3D coordinates (backbone + every
  side-chain atom), built with standard bond lengths/angles via internal
  coordinates (not a relaxed/minimized structure — like the built-in
  phosphorylation patches' geometry, this is approximate and relies on the
  pipeline's own minimization stage to relax it).

No `params.itp` is included, since (as above) every atom type already exists
in the base force field.

## Try it

```bash
# Check the bundle is internally consistent (atom names match across files,
# charges sum to an integer) without touching GROMACS:
phosp validate-ncaa-bundle examples/ncaa_homoallylglycine/bundle

# Check the full config (also lints the bundle, plus the usual tool checks):
phosp validate examples/ncaa_homoallylglycine/run.yaml

# Run it (requires GROMACS + charmm36m-jul2022 installed):
phosp run examples/ncaa_homoallylglycine/run.yaml
```

`phosp run` will graft `HAG` onto residue 48 (superposing the template's
backbone onto the real one via Kabsch alignment), then build a per-run
force-field directory with `bundle/residue.rtp`/`residue.hdb` merged in
before calling `pdb2gmx`.
