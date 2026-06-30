from phosp.modification.base import Modifier


class PThrModifier(Modifier):
    phospho_type = "pThr"

    def _get_bridging_atom_name(self) -> str:
        return "OG1"
