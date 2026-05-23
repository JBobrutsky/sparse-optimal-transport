from importlib.metadata import PackageNotFoundError, version

from .emd import emd, emd2
from sparse_ot.feasibility import InfeasibleProblemError  # noqa: F401

try:
    __version__ = version("sparse-ot")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["emd", "emd2", "InfeasibleProblemError"]
