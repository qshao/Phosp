from phosp.modification.base import Modifier


class PTyrModifier(Modifier):
    phospho_type = "pTyr"

    def _get_bridging_atom_name(self) -> str:
        return "OH"
