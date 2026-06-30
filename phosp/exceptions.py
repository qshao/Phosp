"""Exception classes for the phosp framework."""


class PhospError(Exception):
    """Base exception for all phosp errors."""
    pass


class StageInputError(PhospError):
    """Raised when a stage receives invalid input."""
    pass


class ModificationError(PhospError):
    """Raised when phosphorylation modification fails."""
    pass


class PreparationError(PhospError):
    """Raised when system preparation fails."""
    pass


class SimulationError(PhospError):
    """Raised when MD simulation fails."""
    pass


class AnalysisError(PhospError):
    """Raised when analysis fails."""
    pass
