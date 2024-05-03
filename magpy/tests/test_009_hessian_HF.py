import psi4
import magpy
import pytest
from ..data.molecules import *
import numpy as np
import os

np.set_printoptions(precision=10, linewidth=200, threshold=200, suppress=True)

def test_Hessian_H2O_STO3G():
    psi4.core.clean_options()
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

    hessian = magpy.Hessian(mol, 0, 1)
    disp = 0.001
    e_conv = 1e-13
    r_conv = 1e-13
    maxiter = 400
    max_diis=8
    start_diis=1
    print_level=1
    hess = hessian.compute('HF', disp, e_conv=e_conv, r_conv=r_conv, maxiter=maxiter, max_diis=max_diis, start_diis=start_diis, print_level=print_level)

    # CFOUR HF/STO-3G analytic Hessian
    hess_ref = np.array([
     [  0.0000002355,        0.0000000000,        0.0000000000],
     [ -0.0000001178,        0.0000000000,        0.0000000000],
     [ -0.0000001178,        0.0000000000,        0.0000000000],
     [  0.0000000000,        0.8039499964,        0.0000000000],
     [  0.0000000000,       -0.4019749982,       -0.3371363249],
     [  0.0000000000,       -0.4019749982,        0.3371363249],
     [  0.0000000000,        0.0000000000,        0.6348993605],
     [  0.0000000000,       -0.2163116072,       -0.3174496803],
     [  0.0000000000,        0.2163116072,       -0.3174496803],
     [ -0.0000001178,        0.0000000000,        0.0000000000],
     [  0.0000001262,        0.0000000000,        0.0000000000],
     [ -0.0000000084,        0.0000000000,        0.0000000000],
     [  0.0000000000,       -0.4019749982,       -0.2163116072],
     [  0.0000000000,        0.4389111622,        0.2767239660],
     [  0.0000000000,       -0.0369361640,       -0.0604123588],
     [  0.0000000000,       -0.3371363249,       -0.3174496803],
     [  0.0000000000,        0.2767239660,        0.3001030221],
     [  0.0000000000,        0.0604123588,        0.0173466582],
     [ -0.0000001178,        0.0000000000,        0.0000000000],
     [ -0.0000000084,        0.0000000000,        0.0000000000],
     [  0.0000001262,        0.0000000000,        0.0000000000],
     [  0.0000000000,       -0.4019749982,        0.2163116072],
     [  0.0000000000,       -0.0369361640,        0.0604123588],
     [  0.0000000000,        0.4389111622,       -0.2767239660],
     [  0.0000000000,        0.3371363249,       -0.3174496803],
     [  0.0000000000,       -0.0604123588,        0.0173466582],
     [  0.0000000000,       -0.2767239660,        0.3001030221]
    ])

    assert(np.max(np.abs(hess-hess_ref.reshape(9,9))) < 1e-6)

