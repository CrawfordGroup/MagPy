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
#mol = psi4.geometry(moldict["H2O"])

#H = Hamiltonian(mol)

#scf = hfwfn(H)
#e_conv = 1e-13
#r_conv = 1e-13
#maxiter = 100
#escf, C = scf.solve_scf(e_conv, r_conv, maxiter)

## Test Electric Dipole Against Psi4

# Add a field to the Hamiltonian and re-compute the wfn
#A = 0.00001
#H.add_field(field='electric-dipole', strength=A)
#escf_pos, C_pos = scf.solve_scf(e_conv, r_conv, maxiter)

# Reset potential to zero field, add new field, and re-compute
#H.reset_V()
#H.add_field(field='electric-dipole', axis='z', strength=-A)
#escf_neg, C_neg = scf.solve_scf(e_conv, r_conv, maxiter)

# Compute electronic contribution to the dipole moment via finite differences: mu_e(alpha) = - dE/dF_alpha
#mu_e = -(escf_pos - escf_neg)/(2 * A)

# Grab the nuclear contribution to the dipole from the molecule object
#mu_n = mol.nuclear_dipole()

#print("ESCF(0)     = %20.12f" % (escf))
#print("ESCF(+A)    = %20.12f" % (escf_pos))
#print("ESCF(-A)    = %20.12f" % (escf_neg))
#print("Mu_e(z)     = %20.12f" % (mu_e))
#print("Mu_n(z)     = %20.12f" % (mu_n[2]))
#print("Mu_tot(z)   = %20.12f" % (mu_e+mu_n[2]))

#rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

#print("Mu_tot(z)   = %20.12f (Psi4)" % (psi4.variable('SCF DIPOLE')[2]))

# Test on HF molecule
psi4.set_options({'basis': 'cc-pVDZ',
                  'scf_type': 'pk',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'r_convergence': 1e-12})
hf = """
0 1
F   0.00  0.00 -0.095196019672231
H   0.00  0.00  1.794530105785769
units bohr
no_reorient
nocom
symmetry c1
"""

mol = psi4.geometry(hf)
H = Hamiltonian(mol)
scf = hfwfn(H)
e_conv = 1e-13
r_conv = 1e-13
maxiter = 100
#escf, C = scf.solve_scf(e_conv, r_conv, maxiter)

A = 0.001
H.add_field(field='electric-dipole', strength=[0.0, 0.0, A])
escf_pos, C_pos = scf.solve_scf(e_conv, r_conv, maxiter)

#H.reset_V()
#H.add_field(field='electric-dipole', strength=[0.0, 0.0, -A])
#escf_neg, C_neg = scf.solve_scf(e_conv, r_conv, maxiter)

#mu_e = -(escf_pos - escf_neg)/(2 * A)
#mu_n = mol.nuclear_dipole()

#print("ESCF(0)     = %20.12f" % (escf))
print("ESCF(+A)    = %20.12f" % (escf_pos))
#print("ESCF(-A)    = %20.12f" % (escf_neg))
#print("Mu_e(z)     = %20.12f" % (mu_e))
#print("Mu_n(z)     = %20.12f" % (mu_n[2]))
#print("Mu_tot(z)   = %20.12f" % (mu_e+mu_n[2]))

#rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

#print("Mu_tot(z)   = %20.12f (Psi4)" % (psi4.variable('SCF DIPOLE')[2]))

