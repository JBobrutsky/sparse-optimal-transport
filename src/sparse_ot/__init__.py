from .emd import emd, emd2
from sparse_ot.feasibility import InfeasibleProblemError  # noqa: F401

__version__ = "0.1.0"
__all__ = ["emd", "emd2", "InfeasibleProblemError"]
