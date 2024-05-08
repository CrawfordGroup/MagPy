import psi4
import magpy
import pytest
from ..data.molecules import *
import numpy as np
import os
from ..utils import make_np_array

np.set_printoptions(precision=10, linewidth=200, threshold=200, suppress=True)

def test_APT_H2O_STO3G():
    psi4.core.clean_options()
    psi4.set_memory('2 GB')
    psi4.set_output_file('output.dat', False)
    psi4.set_options({'scf_type': 'pk',
                      'e_convergence': 1e-13,
                      'd_convergence': 1e-13,
                      'r_convergence': 1e-13})

    psi4.set_options({'basis': 'STO-3G'})
    # MP2/STO-3G Optimized geometry from CFOUR
    mol = psi4.geometry("""
O  0.000000000000000  -0.000000000000000   0.141635981802241
H  0.000000000000000  -1.437405158329807  -1.123932904416828
H  0.000000000000000   1.437405158329807  -1.123932904416828
no_com
no_reorient
symmetry c1
units bohr
            """)

    apt = magpy.APT(mol)
    dipder = apt.compute('MP2')

    # CFOUR MP2/STO-3G analytic dipole derivatives
    dipder_ref = np.array([
 [ -0.4973007622,  0.0000000000,  0.0000000000],
 [  0.0000000000,  0.0409050306,  0.0000000000],
 [  0.0000000000,  0.0000000000,  0.1657356778],
 [  0.2486503811,  0.0000000000,  0.0000000000],
 [  0.0000000000, -0.0204525153, -0.1673618446],
 [  0.0000000000, -0.2369326775, -0.0828678389],
 [  0.2486503811,  0.0000000000,  0.0000000000],
 [  0.0000000000, -0.0204525153,  0.1673618446],
 [  0.0000000000,  0.2369326775, -0.0828678389]
    ])

    assert(np.max(np.abs(dipder-dipder_ref)) < 1e-5)

