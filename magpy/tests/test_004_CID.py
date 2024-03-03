import psi4
import magpy
import pytest
from ..data.molecules import *
import numpy as np
import os

np.set_printoptions(precision=10, linewidth=200, threshold=200, suppress=True)

def test_CID_H2O_STO3G():

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
    print(f"  Frozen core = {psi4.core.get_global_option('freeze_core')}")

    psi4.set_options({'freeze_core': 'false'})
    H = magpy.Hamiltonian(mol)
    scf = magpy.hfwfn(H)
    e_conv = 1e-13
    r_conv = 1e-13
    maxiter = 100
    escf, C = scf.solve_scf(e_conv, r_conv, maxiter, print=1)

    cid = magpy.ciwfn(scf)
    eci, C2 = cid.solve_cid(e_conv, r_conv, maxiter)

    c4scf = -74.94207992819220
    c4ci = -0.06865825074438
    assert(abs(escf - c4scf) < 1e-11)
    assert(abs(eci - c4ci) < 1e-11)

    psi4.set_options({'freeze_core': 'true'})
    H = magpy.Hamiltonian(mol)
    scf = magpy.hfwfn(H)
    e_conv = 1e-13
    r_conv = 1e-13
    maxiter = 100
    escf, C = scf.solve_scf(e_conv, r_conv, maxiter, print=1)

    cid = magpy.ciwfn(scf)
    eci, C2 = cid.solve_cid(e_conv, r_conv, maxiter)

    c4scf = -74.94207992819220
    c4ci = -0.06859643905558
    assert(abs(escf - c4scf) < 1e-11)
    assert(abs(eci - c4ci) < 1e-11)


def test_CID_H2O_CCPVDZ():

    psi4.set_memory('2 GB')
    psi4.set_output_file('output.dat', False)
    psi4.set_options({'scf_type': 'pk',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12,
                      'r_convergence': 1e-12})

    psi4.set_options({'basis': 'cc-pVDZ'})
    mol = psi4.geometry(moldict["H2O"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
    print(f"  SCF Energy from Psi4: {rhf_e}")

    psi4.set_options({'freeze_core': 'false'})
    H = magpy.Hamiltonian(mol)
    scf = magpy.hfwfn(H)
    e_conv = 1e-13
    r_conv = 1e-13
    maxiter = 100
    escf, C = scf.solve_scf(e_conv, r_conv, maxiter, print=1)

    cid = magpy.ciwfn(scf)
    eci, C2 = cid.solve_cid(e_conv, r_conv, maxiter, alg='PROJECTED')

    # Compae to CFOUR results
    c4scf = -75.98979581991861
    c4ci = -0.21279410950205
    assert(abs(escf - c4scf) < 1e-11)
    assert(abs(eci - c4ci) < 1e-11)

    psi4.set_options({'freeze_core': 'true'})
    H = magpy.Hamiltonian(mol)
    scf = magpy.hfwfn(H)
    e_conv = 1e-13
    r_conv = 1e-13
    maxiter = 100
    escf, C = scf.solve_scf(e_conv, r_conv, maxiter, print=1)

    cid = magpy.ciwfn(scf)
    eci, C2 = cid.solve_cid(e_conv, r_conv, maxiter)

    c4scf = -75.98979581991861
    c4ci = -0.21098966441656
    assert(abs(escf - c4scf) < 1e-11)
    assert(abs(eci - c4ci) < 1e-11)
