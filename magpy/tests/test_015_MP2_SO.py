import psi4
import magpy
import pytest
from ..data.molecules import *
import numpy as np
import os

np.set_printoptions(precision=10, linewidth=200, threshold=200, suppress=True)

def test_MP2_SO_H2O_STO3G():
    psi4.core.clean_options()
    psi4.core.clean_options()
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
    scf = magpy.hfwfn(H, 0, 1)
    e_conv = 1e-13
    r_conv = 1e-13
    escf, C = scf.solve(e_conv=e_conv, r_conv=r_conv, print_level=1)

    mp2 = magpy.mpwfn_so(scf)
    emp2, C0, C2 = mp2.solve(normalization='intermediate', print_level=1)
    c4mp2 = -0.049149636120
    C0_ref = 1.0
    assert(abs(escf - rhf_e) < 1e-11)
    assert(abs(emp2 - c4mp2) < 1e-11)
    assert(abs(C0 - C0_ref) < 1e-11)

    mp2 = magpy.mpwfn_so(scf)
    emp2, C0, C2 = mp2.solve(normalization='full', print_level=1)
    c4mp2 = -0.049149636120
    C0_ref = 0.9891827770673378
    assert(abs(escf - rhf_e) < 1e-11)
    assert(abs(emp2 - c4mp2) < 1e-11)
    assert(abs(C0 - C0_ref) < 1e-11)

    # Test frozen core
    psi4.set_options({'freeze_core': 'true'})
    H = magpy.Hamiltonian(mol)
    scf = magpy.hfwfn(H, 0, 1)
    e_conv = 1e-13
    r_conv = 1e-13
    escf, C = scf.solve(e_conv=e_conv, r_conv=r_conv, print_level=1)

    mp2 = magpy.mpwfn_so(scf)
    emp2, C0, C2 = mp2.solve(normalization='intermediate', print_level=1)
    c4mp2 = -0.049060280876
    C0_ref = 1.0
    assert(abs(escf - rhf_e) < 1e-11)
    assert(abs(emp2 - c4mp2) < 1e-11)
    assert(abs(C0 - C0_ref) < 1e-11)

    mp2 = magpy.mpwfn_so(scf)
    emp2, C0, C2 = mp2.solve(normalization='full', print_level=1)
    c4mp2 = -0.049060280876
    C0_ref = 0.9891844557765184
    assert(abs(escf - rhf_e) < 1e-11)
    assert(abs(emp2 - c4mp2) < 1e-11)
    assert(abs(C0 - C0_ref) < 1e-11)


def test_MP2_SO_H2O_CCPVDZ():
    psi4.core.clean_options()
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
    scf = magpy.hfwfn(H, 0, 1)
    e_conv = 1e-13
    r_conv = 1e-13
    escf, C = scf.solve(e_conv=e_conv, r_conv=r_conv, print_level=1)

    mp2 = magpy.mpwfn_so(scf)
    emp2, C0, C2 = mp2.solve(normalization='intermediate', print_level=1)
    c4mp2 = -0.214347601415
    C0_ref = 1.0
    assert(abs(escf - rhf_e) < 1e-11)
    assert(abs(emp2 - c4mp2) < 1e-11)
    assert(abs(C0 - C0_ref) < 1e-11)

    mp2 = magpy.mpwfn_so(scf)
    emp2, C0, C2 = mp2.solve(normalization='full', print_level=1)
    c4mp2 = -0.214347601415
    C0_ref = 0.9716969060925619
    assert(abs(escf - rhf_e) < 1e-11)
    assert(abs(emp2 - c4mp2) < 1e-11)
    assert(abs(C0 - C0_ref) < 1e-11)

    psi4.set_options({'freeze_core': 'true'})
    H = magpy.Hamiltonian(mol)
    scf = magpy.hfwfn(H, 0, 1)
    e_conv = 1e-13
    r_conv = 1e-13
    escf, C = scf.solve(e_conv=e_conv, r_conv=r_conv, print_level=1)

    mp2 = magpy.mpwfn_so(scf)
    emp2, C0, C2 = mp2.solve(normalization='intermediate', print_level=1)
    c4mp2 = -0.212229959873
    C0_ref = 1.0
    assert(abs(escf - rhf_e) < 1e-11)
    assert(abs(emp2 - c4mp2) < 1e-11)
    assert(abs(C0 - C0_ref) < 1e-11)

    mp2 = magpy.mpwfn_so(scf)
    emp2, C0, C2 = mp2.solve(normalization='full', print_level=1)
    c4mp2 = -0.212229959873
    C0_ref = 0.9717330625421687
    assert(abs(escf - rhf_e) < 1e-11)
    assert(abs(emp2 - c4mp2) < 1e-11)
    assert(abs(C0 - C0_ref) < 1e-11)

