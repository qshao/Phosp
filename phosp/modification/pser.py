from phosp.modification.base import Modifier


class PSerModifier(Modifier):
    phospho_type = "pSer"

    def _get_bridging_atom_name(self) -> str:
        return "OG"
