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

        # Loop over magnetic field displacements and store wave functions (six total)
        B_pos = []
        B_neg = []
        for B in range(3):
            strength = np.zeros(3)

            # +B displacement
            H.reset_V()
            strength[B] = B_disp
            H.add_field(field='magnetic-dipole', strength=strength)
            scf = magpy.hfwfn(H, self.charge, self.spin)
            scf.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
            scf.match_phase(scf0)
            B_pos.append(scf)

            # -B displacement
            H.reset_V()
            strength[B] = -B_disp
            H.add_field(field='magnetic-dipole', strength=strength)
            scf = magpy.hfwfn(H, self.charge, self.spin)
            scf.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
            scf.match_phase(scf0)
            B_neg.append(scf)

        # Loop over atomic coordinate displacements
        R_pos = []
        R_neg = []
        for R in range(3*mol.natom()):

            # +R displacement
            H_pos = magpy.Hamiltonian(shift_geom(mol, R, R_disp))
            scf = magpy.hfwfn(H_pos, self.charge, self.spin)
            scf.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
            scf.match_phase(scf0)
            R_pos.append(scf)

            # -R displacement
            H_neg = magpy.Hamiltonian(shift_geom(mol, R, -R_disp))
            scf = magpy.hfwfn(H_neg, self.charge, self.spin)
            scf.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
            scf.match_phase(scf0)
            R_neg.append(scf)

        S = [[[0 for k in range(4)] for j in range(3)] for i in range(3*mol.natom())] # list of overlap matrices
        for R in range(3*mol.natom()):
            R_pos_C = R_pos[R].C
            R_neg_C = R_neg[R].C
            R_pos_H = R_pos[R].H.basisset
            R_neg_H = R_neg[R].H.basisset
            for B in range(3):
                B_pos_C = B_pos[B].C
                B_neg_C = B_neg[B].C
                B_pos_H = B_pos[B].H.basisset
                B_neg_H = B_neg[B].H.basisset

                S[R][B][0] = mo_overlap(R_pos_C, R_pos_H, B_pos_C, B_pos_H)
                S[R][B][1] = mo_overlap(R_pos_C, R_pos_H, B_neg_C, B_neg_H)
                S[R][B][2] = mo_overlap(R_neg_C, R_neg_H, B_pos_C, B_pos_H)
                S[R][B][3] = mo_overlap(R_neg_C, R_neg_H, B_neg_C, B_neg_H)

        # Occupied MO slice
        o = slice(0,scf0.ndocc)

        AAT = np.zeros((3*mol.natom(), 3))
        for R in range(3*mol.natom()):
            for B in range(3):

                pp = np.linalg.det(S[R][B][0][o,o])
                pm = np.linalg.det(S[R][B][1][o,o])
                mp = np.linalg.det(S[R][B][2][o,o])
                mm = np.linalg.det(S[R][B][3][o,o])

                # Compute AAT element
                AAT[R,B] = 2*(((pp - pm - mp + mm).imag)/(4*R_disp*B_disp))

        return AAT
