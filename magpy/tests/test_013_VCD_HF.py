import psi4
import magpy
import pytest
from ..data.molecules import *
import numpy as np
import os

np.set_printoptions(precision=10, linewidth=200, threshold=200, suppress=True)

def test_VCD_H2O_STO3G():

    psi4.set_memory('2 GB')
    psi4.set_output_file('output.dat', False)
    psi4.set_options({'scf_type': 'pk',
                      'e_convergence': 1e-13,
                      'd_convergence': 1e-13,
                      'r_convergence': 1e-13})

    psi4.set_options({'basis': 'STO-3G'})
    # HF/STO-3G Optimized geometry from CFOUR
    mol = psi4.geometry("""
O  0.000000000000000  -0.000000000000000   0.134464865292648
H  0.000000000000000  -1.432565142139463  -1.067027493065974
H  0.000000000000000   1.432565142139463  -1.067027493065974
no_com
no_reorient
symmetry c1
units bohr
            """)

    magpy.normal(mol)
