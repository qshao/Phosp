from phosp.modification.phospho import PhosphoModifier


class PTyrModifier(PhosphoModifier):
    mod_type = "pTyr"
    ff_resnames = {"charmm36m": "PTR", "amber_ff14sb": "PTR"}

    def _get_bridging_atom_name(self) -> str:
        return "OH"
