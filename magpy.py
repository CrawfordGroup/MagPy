import psi4
import sys
sys.path.append('/Users/crawdad/src/pycc/pycc/data')
from molecules import *
from aat_hf import AAT_HF

psi4.set_memory('2 GB')
psi4.set_output_file('output.dat', False)
psi4.set_options({'basis': 'STO-3G',
                  'scf_type': 'pk',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'r_convergence': 1e-12})
mol = psi4.geometry(moldict["H2O"])
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

print(f"  SCF Energy from Psi4: {rhf_e}")

AAT = AAT_HF(mol)

r_disp = 0.001
b_disp = 0.001
I = AAT.compute(r_disp, b_disp)
