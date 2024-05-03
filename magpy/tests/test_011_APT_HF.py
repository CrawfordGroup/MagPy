import psi4
import magpy
import pytest
from ..data.molecules import *
import numpy as np
import os
from ..utils import make_np_array

np.set_printoptions(precision=10, linewidth=200, threshold=200, suppress=True)

def test_APT_H2O_STO3G():

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

    apt = magpy.APT(mol)
    R_disp = 0.001
    F_disp = 0.0001
    e_conv = 1e-14
    r_conv = 1e-14
    maxiter = 400
    max_diis=8
    start_diis=1
    print_level=1
    dipder = apt.compute('HF', R_disp, F_disp, e_conv=e_conv, r_conv=r_conv, maxiter=maxiter, max_diis=max_diis, start_diis=start_diis, print_level=print_level)
    print(dipder)

    # CFOUR HF/STO-3G analytic dipole derivatives
    dipder_ref = make_np_array("""
[[-0.5596822336  0.            0.          ]
 [ 0.           -0.0311494713  0.          ]
 [ 0.            0.            0.0835319998]
 [ 0.2798411169  0.            0.          ]
 [ 0.            0.0155747356 -0.1570627446]
 [ 0.           -0.2216402091 -0.0417659999]
 [ 0.2798411169  0.            0.          ]
 [ 0.            0.0155747356  0.1570627446]
 [ 0.            0.2216402091 -0.0417659999]]
    """)

    assert(np.max(np.abs(dipder-dipder_ref)) < 1e-5)

