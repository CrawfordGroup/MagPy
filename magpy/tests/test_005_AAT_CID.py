import psi4
import magpy
import pytest
from ..data.molecules import *
import numpy as np
import os

np.set_printoptions(precision=10, linewidth=200, threshold=200, suppress=True)

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

    AAT = magpy.AAT_CI(mol)

    r_disp = 0.0001
    b_disp = 0.0001
    I_00, I_0D, I_D0 = AAT.compute(r_disp, b_disp)
    print("\nElectronic Contribution to Atomic Axial Tensor (a.u.):")
    print(I_00)
    print(I_0D)
    print(I_D0)
    print(I_0D+I_D0)

