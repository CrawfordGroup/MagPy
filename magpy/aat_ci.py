if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")

import psi4
import magpy
import numpy as np
from .utils import *
import pickle


class AAT_CI(object):

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
        ci0 = magpy.ciwfn(scf0) # Not strictly necessary, but handy

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
            ci = magpy.ciwfn(scf)
            ci.solve_cid(e_conv, r_conv, maxiter, max_diis, start_diis)
            B_pos.append(ci)

            # -B displacement
            H.reset_V()
            strength[B] = -B_disp
            H.add_field(field='magnetic-dipole', strength=strength)
            scf = magpy.hfwfn(H, self.charge, self.spin)
            scf.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
            scf.match_phase(scf0)
            ci = magpy.ciwfn(scf)
            ci.solve_cid(e_conv, r_conv, maxiter, max_diis, start_diis)
            B_neg.append(ci)

        # Loop over atomic coordinate displacements
        R_pos = []
        R_neg = []
        for R in range(3*mol.natom()):

            # +R displacement
            H = magpy.Hamiltonian(shift_geom(mol, R, R_disp))
            scf = magpy.hfwfn(H, self.charge, self.spin)
            scf.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
            scf.match_phase(scf0)
            ci = magpy.ciwfn(scf)
            ci.solve_cid(e_conv, r_conv, maxiter, max_diis, start_diis)
            R_pos.append(ci)

            # -R displacement
            H = magpy.Hamiltonian(shift_geom(mol, R, -R_disp))
            scf = magpy.hfwfn(H, self.charge, self.spin)
            scf.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
            scf.match_phase(scf0)
            ci = magpy.ciwfn(scf)
            ci.solve_cid(e_conv, r_conv, maxiter, max_diis, start_diis)
            R_neg.append(ci)

        # Compute full MO overlap matrix for all combinations of perturbed MOs
        S = [[[0 for k in range(4)] for j in range(3)] for i in range(3*mol.natom())] # list of overlap matrices
        for R in range(3*mol.natom()):
            R_pos_C = R_pos[R].hfwfn.C
            R_neg_C = R_neg[R].hfwfn.C
            R_pos_H = R_pos[R].hfwfn.H.basisset
            R_neg_H = R_neg[R].hfwfn.H.basisset
            for B in range(3):
                B_pos_C = B_pos[B].hfwfn.C
                B_neg_C = B_neg[B].hfwfn.C
                B_pos_H = B_pos[B].hfwfn.H.basisset
                B_neg_H = B_neg[B].hfwfn.H.basisset

                S[R][B][0] = mo_overlap(R_pos_C, R_pos_H, B_pos_C, B_pos_H)
                S[R][B][1] = mo_overlap(R_pos_C, R_pos_H, B_neg_C, B_neg_H)
                S[R][B][2] = mo_overlap(R_neg_C, R_neg_H, B_pos_C, B_pos_H)
                S[R][B][3] = mo_overlap(R_neg_C, R_neg_H, B_neg_C, B_neg_H)

        # Compute AAT components using finite-difference
        o = slice(0,scf0.ndocc)
        no = ci0.no
        nv = ci0.nv

        # <d0/dR|d0/dB>
        AAT_00 = np.zeros((3*mol.natom(), 3))
        for R in range(3*mol.natom()):
            for B in range(3):

                pp = np.linalg.det(S[R][B][0][o,o])
                pm = np.linalg.det(S[R][B][1][o,o])
                mp = np.linalg.det(S[R][B][2][o,o])
                mm = np.linalg.det(S[R][B][3][o,o])

                # Compute AAT element
                AAT_00[R,B] = 2*(((pp - pm - mp + mm)/(4*R_disp*B_disp))).imag

        AAT_0D = np.zeros((3*mol.natom(), 3))
        AAT_D0 = np.zeros((3*mol.natom(), 3))
        for R in range(3*mol.natom()):
            ci_R_pos = R_pos[R]
            ci_R_neg = R_neg[R]

            for B in range(3):
                ci_B_pos = B_pos[B]
                ci_B_neg = B_neg[B]

                # <d0/dR|dD/dB>
                pp = pm = mp = mm = 0.0
                for i in range(no):
                    for a in range(nv):
                        for j in range(no):
                            for b in range(nv):

                                det_AA = det_overlap([0], 'AA', [i, a+no, j, b+no], 'AA', S[R][B][0], o)
                                det_AB = det_overlap([0], 'AB', [i, a+no, j, b+no], 'AB', S[R][B][0], o)
                                pp += 0.5 * (ci_B_pos.C2[i,j,a,b] - ci_B_pos.C2[i,j,b,a]) * det_AA + ci_B_pos.C2[i,j,a,b] * det_AB

                                det_AA = det_overlap([0], 'AA', [i, a+no, j, b+no], 'AA', S[R][B][1], o)
                                det_AB = det_overlap([0], 'AB', [i, a+no, j, b+no], 'AB', S[R][B][1], o)
                                pm += 0.5 * (ci_B_neg.C2[i,j,a,b] - ci_B_neg.C2[i,j,b,a]) * det_AA + ci_B_neg.C2[i,j,a,b] * det_AB

                                det_AA = det_overlap([0], 'AA', [i, a+no, j, b+no], 'AA', S[R][B][2], o)
                                det_AB = det_overlap([0], 'AB', [i, a+no, j, b+no], 'AB', S[R][B][2], o)
                                mp += 0.5 * (ci_B_pos.C2[i,j,a,b] - ci_B_pos.C2[i,j,b,a]) * det_AA + ci_B_pos.C2[i,j,a,b] * det_AB

                                det_AA = det_overlap([0], 'AA', [i, a+no, j, b+no], 'AA', S[R][B][3], o)
                                det_AB = det_overlap([0], 'AB', [i, a+no, j, b+no], 'AB', S[R][B][3], o)
                                mm += 0.5 * (ci_B_neg.C2[i,j,a,b] - ci_B_neg.C2[i,j,b,a]) * det_AA + ci_B_neg.C2[i,j,a,b] * det_AB

                AAT_0D[R,B] = (((pp - pm - mp + mm)/(4*R_disp*B_disp))).imag

                # <dD/dR|d0/dB>
                pp = pm = mp = mm = 0.0
                for i in range(no):
                    for a in range(nv):
                        for j in range(no):
                            for b in range(nv):

                                det_AA = det_overlap([i, a+no, j, b+no], 'AA', [0], 'AA', S[R][B][0], o)
                                det_AB = det_overlap([i, a+no, j, b+no], 'AB', [0], 'AB', S[R][B][0], o)
                                pp += 0.5 * (ci_R_pos.C2[i,j,a,b] - ci_R_pos.C2[i,j,b,a]) * det_AA + ci_R_pos.C2[i,j,a,b] * det_AB
            
                                det_AA = det_overlap([i, a+no, j, b+no], 'AA', [0], 'AA', S[R][B][1], o)
                                det_AB = det_overlap([i, a+no, j, b+no], 'AB', [0], 'AB', S[R][B][1], o)
                                pm += 0.5 * (ci_R_neg.C2[i,j,a,b] - ci_R_neg.C2[i,j,b,a]) * det_AA + ci_R_neg.C2[i,j,a,b] * det_AB
            
                                det_AA = det_overlap([i, a+no, j, b+no], 'AA', [0], 'AA', S[R][B][2], o)
                                det_AB = det_overlap([i, a+no, j, b+no], 'AB', [0], 'AB', S[R][B][2], o)
                                mp += 0.5 * (ci_R_pos.C2[i,j,a,b] - ci_R_pos.C2[i,j,b,a]) * det_AA + ci_R_pos.C2[i,j,a,b] * det_AB
            
                                det_AA = det_overlap([i, a+no, j, b+no], 'AA', [0], 'AA', S[R][B][3], o)
                                det_AB = det_overlap([i, a+no, j, b+no], 'AB', [0], 'AB', S[R][B][3], o)
                                mm += 0.5 * (ci_R_neg.C2[i,j,a,b] - ci_R_neg.C2[i,j,b,a]) * det_AA + ci_R_neg.C2[i,j,a,b] * det_AB

                AAT_D0[R,B] = (((pp - pm - mp + mm)/(4*R_disp*B_disp))).imag

        return AAT_00, AAT_0D, AAT_D0
