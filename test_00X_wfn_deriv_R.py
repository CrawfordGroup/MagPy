import psi4
import sys
sys.path.append('/Users/crawdad/src/pycc/pycc/data')
from molecules import *
from hamiltonian import Hamiltonian
from hfwfn import hfwfn
import numpy as np
import qcelemental as qcel

psi4.set_memory('2 GB')
psi4.set_output_file('output.dat', False)
psi4.set_options({'basis': 'STO-3G',
                  'scf_type': 'pk',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'r_convergence': 1e-12})
molecule = psi4.geometry(moldict["H2O_Teach"])
molecule.fix_orientation(True)
molecule.fix_com(True)
molecule.update_geometry()
geom, mass, elem, elez, uniq = molecule.to_arrays()
print(geom)

#for at in range(molecule.natom()):
#    molecule.atoms[at]


#H = Hamiltonian(molecule)

# Displace geometry and run new SCF for each Cartesian coordinate
disp = +0.0025
geom[0,2] = geom[0,2] + disp
molecule.set_geometry(psi4.core.Matrix.from_array(geom.reshape(-1,3)))
print(geom)

geom, mass, elem, elez, uniq = molecule.to_arrays()
print(geom)

#for i in range(len(mass)):
#    for j in range(3):
#        geom[i,j])


#print(molecule.to_arrays()[0])

