import psi4
import magpy
import pytest
from ..data.molecules import *
import numpy as np
import os

np.set_printoptions(precision=10, linewidth=200, threshold=200, suppress=True)

def test_VCD_H2O2_STO3GP():

    psi4.set_memory('2 GB')
    psi4.set_output_file('output.dat', False)
    psi4.set_options({'scf_type': 'pk',
                      'e_convergence': 1e-13,
                      'd_convergence': 1e-13,
                      'r_convergence': 1e-13})

    psi4.set_options({'basis': 'STO-3G'})
    # CFOUR HF/STO-3G optimized geometry
    mol = psi4.geometry("""
O       0.0000000000        1.3192641900       -0.0952542913
O      -0.0000000000       -1.3192641900       -0.0952542913
H       1.6464858700        1.6841036400        0.7620343300
H      -1.6464858700       -1.6841036400        0.7620343300
no_com
no_reorient
symmetry c1
units bohr
            """)

    freq, ir, vcd = magpy.normal(mol, 'HF')

    # CFOUR
    freq_ref = np.array([4148.2766, 4140.8893, 1781.0495, 1589.6394, 1486.9510, 184.6327])
    # CFOUR
    ir_ref = np.array([30.3023, 12.7136, 2.0726, 46.0798, 0.0168, 134.1975])
    # DALTON (*-1 because it appears they're rotatory strengths have the wrong sign)
    vcd_ref = np.array([50.538, -53.528, 28.607, -14.650, 1.098, 101.378])

    assert(np.max(np.abs(freq-freq_ref)) < 0.1)
    assert(np.max(np.abs(ir-ir_ref)) < 0.1)
    assert(np.max(np.abs(vcd-vcd_ref)) < 0.1)

