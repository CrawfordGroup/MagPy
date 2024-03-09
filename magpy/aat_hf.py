if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")

import psi4
import magpy
import numpy as np
from .utils import *


class AAT_HF(object):

    def __init__(self, molecule, charge=0, spin=1):

        # Ensure geometry remains fixed in space
        molecule.fix_orientation(True)
        molecule.fix_com(True)
        molecule.update_geometry()
        self.molecule = molecule

        self.charge = charge
        self.spin = spin


    def compute(self, R_disp, B_disp, e_conv=1e-10, r_conv=1e-10, maxiter=400, max_diis=8, start_diis=1, print_level=0):

        mol = self.molecule

        # Unperturbed Hamiltonian
        H = magpy.Hamiltonian(mol)

        # Compute the unperturbed HF wfn
        scf0 = magpy.hfwfn(H, self.charge, self.spin)
        scf0.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)

        # Occupied MO slice
        o = slice(0,scf0.ndocc)

        AAT = np.zeros((3*mol.natom(), 3))

        # Loop over magnetic field displacements and store wave functions (six total)
        scf_B_pos = []
        scf_B_neg = []
        for B in range(3):
            strength = np.zeros(3)

            # +B displacement
            H.reset_V()
            strength[B] = B_disp
            H.add_field(field='magnetic-dipole', strength=strength)
            scf_B = magpy.hfwfn(H, self.charge, self.spin)
            scf_B.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
            scf_B.match_phase(scf0)
            scf_B_pos.append(scf_B)

            # -B displacement
            H.reset_V()
            strength[B] = -B_disp
            H.add_field(field='magnetic-dipole', strength=strength)
            scf_B = magpy.hfwfn(H, self.charge, self.spin)
            scf_B.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
            scf_B.match_phase(scf0)
            scf_B_neg.append(scf_B)

        # Loop over atomic coordinate displacements
        for R in range(3*mol.natom()):

            # +R displacement
            H_pos = magpy.Hamiltonian(shift_geom(mol, R, R_disp))
            scf_R_pos = magpy.hfwfn(H_pos, self.charge, self.spin)
            scf_R_pos.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
            scf_R_pos.match_phase(scf0)

            # -R displacement
            H_neg = magpy.Hamiltonian(shift_geom(mol, R, -R_disp))
            scf_R_neg = magpy.hfwfn(H_neg, self.charge, self.spin)
            scf_R_neg.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
            scf_R_neg.match_phase(scf0)

            # Compute determinantal overlaps for finite-difference
            for B in range(3):
                pp = det_overlap(scf_R_pos.C[:,o], scf_R_pos.H.basisset, scf_B_pos[B].C[:,o], scf_B_pos[B].H.basisset)
                pm = det_overlap(scf_R_pos.C[:,o], scf_R_pos.H.basisset, scf_B_neg[B].C[:,o], scf_B_neg[B].H.basisset)
                mp = det_overlap(scf_R_neg.C[:,o], scf_R_neg.H.basisset, scf_B_pos[B].C[:,o], scf_B_pos[B].H.basisset)
                mm = det_overlap(scf_R_neg.C[:,o], scf_R_neg.H.basisset, scf_B_neg[B].C[:,o], scf_B_neg[B].H.basisset)

                # Compute AAT element
                AAT[R,B] = 2*(((pp - pm - mp + mm).imag)/(4*R_disp*B_disp))

        return AAT
