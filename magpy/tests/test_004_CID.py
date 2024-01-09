import psi4
import magpy
import pytest
from ..data.molecules import *
import numpy as np
import os

np.set_printoptions(precision=10, linewidth=200, threshold=200, suppress=True)

def test_CID_H2O():

    psi4.set_memory('2 GB')
    psi4.set_output_file('output.dat', False)
    psi4.set_options({'scf_type': 'pk',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12,
                      'r_convergence': 1e-12})

    psi4.set_options({'basis': 'STO-3G'})
    mol = psi4.geometry(moldict["H2O"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
    print(f"  SCF Energy from Psi4: {rhf_e}")

    H = magpy.Hamiltonian(mol)
    scf = magpy.hfwfn(H)
    e_conv = 1e-13
    r_conv = 1e-13
    maxiter = 100
    escf, C = scf.solve_scf(e_conv, r_conv, maxiter, print=1)

    cid = magpy.ciwfn(scf)
    cid.solve_cid(e_conv, r_conv, maxiter)

