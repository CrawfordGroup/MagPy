"""
MagPy: A Python reference implementation for including explicit magnetic fields in quantum chemical calculations.
"""

# Add imports here
from .hamiltonian import Hamiltonian
from .hfwfn import hfwfn
from .ciwfn import ciwfn
from .ciwfn_so import ciwfn_so
from .aat_hf import AAT_HF
from .aat_ci import AAT_CI
from .aat_ci_so import AAT_CI_SO
from .hessian import Hessian
from .apt import APT
from .normal import normal
from .aat import AAT


#from ._version import __version__
