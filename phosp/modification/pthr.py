from phosp.modification.phospho import PhosphoModifier


class PThrModifier(PhosphoModifier):
    mod_type = "pThr"
    ff_resnames = {"charmm36m": "TPO", "amber_ff14sb": "TPO"}

    def _get_bridging_atom_name(self) -> str:
        return "OG1"
