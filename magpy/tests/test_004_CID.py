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

    psi4.set_options({'freeze_core': 'false'})
    H = magpy.Hamiltonian(mol)
    scf = magpy.hfwfn(H, 0, 1, 1)
    e_conv = 1e-13
    r_conv = 1e-13
    maxiter = 100
    escf, C = scf.solve_scf(e_conv, r_conv, maxiter)

    cid = magpy.ciwfn(scf, normalization = 'intermediate')
    eci, C0, C2 = cid.solve_cid(e_conv, r_conv, maxiter)
    c4scf = -74.94207992819220
    c4ci = -0.06865825074438
    C0_ref = 1.0
    assert(abs(escf - c4scf) < 1e-11)
    assert(abs(eci - c4ci) < 1e-11)
    assert(abs(C0 - C0_ref) < 1e-11)

    cid = magpy.ciwfn(scf, normalization = 'full')
    eci, C0, C2 = cid.solve_cid(e_conv, r_conv, maxiter)
    c4scf = -74.94207992819220
    c4ci = -0.06865825074438
    C0_ref = 0.978084764153443
    assert(abs(escf - c4scf) < 1e-11)
    assert(abs(eci - c4ci) < 1e-11)
    assert(abs(C0 - C0_ref) < 1e-11)

    psi4.set_options({'freeze_core': 'true'})
    H = magpy.Hamiltonian(mol)
    scf = magpy.hfwfn(H, 0, 1, 1)
    e_conv = 1e-13
    r_conv = 1e-13
    maxiter = 100
    escf, C = scf.solve_scf(e_conv, r_conv, maxiter)

    cid = magpy.ciwfn(scf, normalization = 'intermediate')
    eci, C0, C2 = cid.solve_cid(e_conv, r_conv, maxiter)
    C0_ref = 1.0
    c4scf = -74.94207992819220
    c4ci = -0.06859643905558
    assert(abs(escf - c4scf) < 1e-11)
    assert(abs(eci - c4ci) < 1e-11)
    assert(abs(C0 - C0_ref) < 1e-11)

    cid = magpy.ciwfn(scf, normalization = 'full')
    eci, C0, C2 = cid.solve_cid(e_conv, r_conv, maxiter)
    C0_ref = 0.9780778514625926
    c4scf = -74.94207992819220
    c4ci = -0.06859643905558
    assert(abs(escf - c4scf) < 1e-11)
    assert(abs(eci - c4ci) < 1e-11)
    assert(abs(C0 - C0_ref) < 1e-11)


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
    scf = magpy.hfwfn(H, 0, 1, 1)
    e_conv = 1e-13
    r_conv = 1e-13
    maxiter = 100
    escf, C = scf.solve_scf(e_conv, r_conv, maxiter)

    cid = magpy.ciwfn(scf, normalization='intermediate')
    eci, C0, C2 = cid.solve_cid(e_conv, r_conv, maxiter)
    C0_ref = 1.0
    c4scf = -75.98979581991861
    c4ci = -0.21279410950205
    assert(abs(escf - c4scf) < 1e-11)
    assert(abs(eci - c4ci) < 1e-11)
    assert(abs(C0 - C0_ref) < 1e-11)

    cid = magpy.ciwfn(scf, normalization='full')
    eci, C0, C2 = cid.solve_cid(e_conv, r_conv, maxiter)
    C0_ref = 0.9712657483891116
    c4scf = -75.98979581991861
    c4ci = -0.21279410950205
    assert(abs(escf - c4scf) < 1e-11)
    assert(abs(eci - c4ci) < 1e-11)
    assert(abs(C0 - C0_ref) < 1e-11)

    psi4.set_options({'freeze_core': 'true'})
    H = magpy.Hamiltonian(mol)
    scf = magpy.hfwfn(H, 0, 1, 1)
    e_conv = 1e-13
    r_conv = 1e-13
    maxiter = 100
    escf, C = scf.solve_scf(e_conv, r_conv, maxiter)

    cid = magpy.ciwfn(scf, normalization='intermediate')
    eci, C0, C2 = cid.solve_cid(e_conv, r_conv, maxiter)
    C0_ref = 1.0
    c4scf = -75.98979581991861
    c4ci = -0.21098966441656
    assert(abs(escf - c4scf) < 1e-11)
    assert(abs(eci - c4ci) < 1e-11)
    assert(abs(C0 - C0_ref) < 1e-11)

    cid = magpy.ciwfn(scf, normalization='full')
    eci, C0, C2 = cid.solve_cid(e_conv, r_conv, maxiter)
    C0_ref = 0.9712377677176858
    c4scf = -75.98979581991861
    c4ci = -0.21098966441656
    assert(abs(escf - c4scf) < 1e-11)
    assert(abs(eci - c4ci) < 1e-11)
    assert(abs(C0 - C0_ref) < 1e-11)

def test_CID_H2O2_STO3G():

    psi4.set_memory('2 GB')
    psi4.set_output_file('output.dat', False)
    psi4.set_options({'scf_type': 'pk',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12,
                      'r_convergence': 1e-12})

    psi4.set_options({'basis': 'STO-3G'})
    mol = psi4.geometry("""
O  0.000000000000000  -0.000000000000000   0.143954618947726
H  0.000000000000000  -1.450386234357036  -1.142332131421532
H  0.000000000000000   1.450386234357036  -1.142332131421532
no_com
no_reorient
symmetry c1
units bohr
            """)

    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
    print(f"  SCF Energy from Psi4: {rhf_e}")

    psi4.set_options({'freeze_core': 'false'})
    H = magpy.Hamiltonian(mol)
    scf = magpy.hfwfn(H, 0, 1, 1)
    e_conv = 1e-13
    r_conv = 1e-13
    maxiter = 100
    escf, C = scf.solve_scf(e_conv, r_conv, maxiter)
    cid = magpy.ciwfn(scf, normalization='intermediate')
    eci, C0, C2 = cid.solve_cid(e_conv, r_conv, maxiter)
