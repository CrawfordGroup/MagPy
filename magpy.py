import psi4
import sys
sys.path.append('/Users/crawdad/src/pycc/pycc/data')
from molecules import *
from hamiltonian import Hamiltonian
from hfwfn import hfwfn
import numpy as np

psi4.set_memory('2 GB')
psi4.set_output_file('output.dat', False)
psi4.set_options({'basis': 'STO-3G',
                  'scf_type': 'pk',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'r_convergence': 1e-12})
mol = psi4.geometry(moldict["H2O"])
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

H = Hamiltonian(mol)

scf = hfwfn(H)
e_conv = 1e-13
r_conv = 1e-13
maxiter = 100
escf, C = scf.solve_scf(e_conv, r_conv, maxiter)

### Test Magnetic Field
A = 0.01
H.reset_V()
H.add_field(field='magnetic-dipole', field_axis='z', strength=A)
escf_mag_pos, C_mag_pos = scf.solve_scf(e_conv, r_conv, maxiter, max_diis=0)
