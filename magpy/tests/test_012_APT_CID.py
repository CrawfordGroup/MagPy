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
    # CID/STO-3G Optimized geometry from CFOUR
    mol = psi4.geometry("""
O  0.000000000000000  -0.000000000000000   0.143954618947726
H  0.000000000000000   1.450386234357036  -1.142332131421532
H  0.000000000000000  -1.450386234357036  -1.142332131421532
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
    dipder = apt.compute('CID', R_disp, F_disp, e_conv=e_conv, r_conv=r_conv, maxiter=maxiter, max_diis=max_diis, start_diis=start_diis, print_level=print_level)

    # Gaussian09 CID/STO-3G finite-difference dipole derivatives
    dipder_ref = np.array([
 [-4.67574414E-01,  7.05005665E-13,  0.00000000E+00],
 [ 8.40982084E-14,  1.07315520E-01, -2.31111593E-33],
 [ 8.36893079E-13,  2.35001888E-13,  1.98997646E-01],
 [ 2.33787312E-01, -8.76562021E-34, -6.16297582E-33],
 [-9.31586489E-14, -5.36578358E-02,  1.77653034E-01],
 [-4.06107292E-13,  2.54923176E-01, -9.94990576E-02],
 [ 2.33787312E-01,  3.52018780E-17,  2.17562220E-17],
 [-9.31234471E-14, -5.36578358E-02, -1.77653034E-01],
 [ 4.06138511E-13, -2.54923176E-01, -9.94990576E-02],
    ])

    assert(np.max(np.abs(dipder-dipder_ref)) < 1e-5)

