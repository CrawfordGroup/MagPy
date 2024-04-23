import psi4
import magpy
import pytest
from ..data.molecules import *
import numpy as np
import os

np.set_printoptions(precision=15, linewidth=200, threshold=200, suppress=True)

def test_AAT_CID_H2DIMER():

    psi4.set_memory('2 GB')
    psi4.set_output_file('output.dat', False)
    psi4.set_options({'scf_type': 'pk',
                      'e_convergence': 1e-13,
                      'd_convergence': 1e-13,
                      'r_convergence': 1e-13})

    psi4.set_options({'basis': 'STO-6G'})
    mol = psi4.geometry(moldict["(H2)_2"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
    print(f"  SCF Energy from Psi4: {rhf_e}")

    AAT = magpy.AAT_CI(mol, 0, 1, 1)

    r_disp = 0.0001
    b_disp = 0.0001
    e_conv = 1e-13
    r_conv = 1e-13
    I_00, I_0D, I_D0, I_DD = AAT.compute(r_disp, b_disp, e_conv, r_conv)
    print("\nElectronic Contribution to Atomic Axial Tensor (a.u.):")
    print("Hartree-Fock component:")
    print(I_00)
    print("<0|D> Component\n")
    print(I_0D)
    print("<D|0> Component\n")
    print(I_D0)
    print("<0|D>+<D|0>\n")
    print(I_0D+I_D0)
    print("<D|D> Component\n")
    print(I_DD)


def test_AAT_CID_H2O():

    psi4.set_memory('2 GB')
    psi4.set_output_file('output.dat', False)
    psi4.set_options({'scf_type': 'pk',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12,
                      'r_convergence': 1e-12})

    psi4.set_options({'basis': 'STO-6G'})
    mol = psi4.geometry(moldict["H2O"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
    print(f"  SCF Energy from Psi4: {rhf_e}")

    AAT = magpy.AAT_CI(mol, 0, 1, 1)

    r_disp = 0.0001
    b_disp = 0.0001
    e_conv = 1e-13
    r_conv = 1e-13
    I_00, I_0D, I_D0, I_DD = AAT.compute(r_disp, b_disp, e_conv, r_conv)
    print("\nElectronic Contribution to Atomic Axial Tensor (a.u.):")
    print("Hartree-Fock component:")
    print(I_00)
    print("<0|D> Component\n")
    print(I_0D)
    print("<D|0> Component\n")
    print(I_D0)
    print("<0|D>+<D|0>\n")
    print(I_0D+I_D0)
    print("<D|D> Component\n")
    print(I_DD)


