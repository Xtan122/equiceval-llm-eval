"""Method adapters: reduce each method's native output to a ``Decision``."""
from expeval.adapters.equiceval_adapter import EquiCEvalAdapter
from expeval.adapters.ra_adapter import RAAdapter

__all__ = ["EquiCEvalAdapter", "RAAdapter"]
