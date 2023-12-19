import psi4
import sys
sys.path.append('data')
from molecules import *
from aat_hf import AAT_HF
import numpy as np

np.set_printoptions(precision=10, linewidth=200, threshold=200, suppress=True)

psi4.set_memory('2 GB')
psi4.set_output_file('output.dat', False)
psi4.set_options({'basis': '4-31G',
                  'scf_type': 'pk',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'r_convergence': 1e-12})

H2O2 = """
    O       0.0000000000        1.3192641900       -0.0952542913
    O      -0.0000000000       -1.3192641900       -0.0952542913
    H       1.6464858700        1.6841036400        0.7620343300
    H      -1.6464858700       -1.6841036400        0.7620343300
symmetry c1
units bohr
no_reorient
nocom
"""

etho = """
O       0.0000000000        0.0000000000        1.6119363900
C       0.0000000000       -1.3813890400       -0.7062143040
C       0.0000000000        1.3813905700       -0.7062514120
H      -1.7489765900       -2.3794725300       -1.1019539000
H       1.7489765900       -2.3794725300       -1.1019539000
H       1.7489765900        2.3794634300       -1.1020178200
H      -1.7489765900        2.3794634300       -1.1020178200
symmetry c1
no_reorient
no_com
units bohr
"""

#mol = psi4.geometry(moldict["H2O_Teach"])
#mol = psi4.geometry(H2O2)
mol = psi4.geometry(etho)
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

print(f"  SCF Energy from Psi4: {rhf_e}")

AAT = AAT_HF(mol)

r_disp = 0.0001
b_disp = 0.0001
I = AAT.compute(r_disp, b_disp)
print("Electronic Contribution to Atomic Axial Tensor (a.u.):")
print(I)
