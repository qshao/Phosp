from phosp.modification.phospho import PhosphoModifier


class PSerModifier(PhosphoModifier):
    mod_type = "pSer"
    ff_resnames = {"charmm36m": "SEP", "amber_ff14sb": "SEP"}

    def _get_bridging_atom_name(self) -> str:
        return "OG"
