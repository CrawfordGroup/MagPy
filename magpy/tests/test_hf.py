import psi4
import sys
sys.path.append('/Users/crawdad/src/pycc/pycc/data')
from molecules import *
from hamiltonian import Hamiltonian
from hfwfn import hfwfn
import numpy as np

psi4.set_memory('2 GB')
psi4.set_output_file('output.dat', False)
psi4.set_options({'basis': 'cc-pVDZ',
                  'scf_type': 'pk',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'r_convergence': 1e-12})

#psi4.set_options({'basis': 'cc-pVDZ',
#                  'scf_type': 'pk',
#                  'e_convergence': 1e-12,
#                  'd_convergence': 1e-12,
#                  'r_convergence': 1e-12})


# Test on HF molecule
hf = """
F 0.0000000 0.0000000 0.0916950
H 0.0000000 0.0000000 -0.8252550
symmetry c1
no_reorient
nocom
"""

mol = psi4.geometry(hf)
#H = Hamiltonian(mol)
#scf = hfwfn(H)
#e_conv = 1e-13
#r_conv = 1e-13
#maxiter = 100
#escf, C = scf.solve_scf(e_conv, r_conv, maxiter)

#A = 0.001
#H.add_field(field='electric-dipole', strength=[0.0, 0.0, A])
#escf_pos, C_pos = scf.solve_scf(e_conv, r_conv, maxiter)

#H.reset_V()
#H.add_field(field='electric-dipole', strength=[0.0, 0.0, -A])
#escf_neg, C_neg = scf.solve_scf(e_conv, r_conv, maxiter)

#mu_e = -(escf_pos - escf_neg)/(2 * A)
#mu_n = mol.nuclear_dipole()

#print("ESCF(0)     = %20.12f" % (escf))
#print("ESCF(+A)    = %20.12f" % (escf_pos))
#print("ESCF(-A)    = %20.12f" % (escf_neg))
#print("Mu_e(z)     = %20.12f" % (mu_e))
#print("Mu_n(z)     = %20.12f" % (mu_n[2]))
#print("Mu_tot(z)   = %20.12f" % (mu_e+mu_n[2]))

rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

#print("Mu_tot(z)   = %20.12f (Psi4)" % (psi4.variable('SCF DIPOLE')[2]))

