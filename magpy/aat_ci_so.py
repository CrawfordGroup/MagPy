if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")

import psi4
import magpy
import numpy as np
from .utils import *


class AAT_CI_SO(object):

    def __init__(self, molecule, charge=0, spin=1, print_level=0):

        # Ensure geometry remains fixed in space
        molecule.fix_orientation(True)
        molecule.fix_com(True)
        molecule.reinterpret_coordentry(False)
        self.molecule = molecule

        self.charge = charge
        self.spin = spin
        self.print_level = print_level


    def compute(self, R_disp, B_disp, e_conv=1e-10, r_conv=1e-10, maxiter=400, max_diis=8, start_diis=1):

        mol = self.molecule

        # Unperturbed Hamiltonian
        H = magpy.Hamiltonian(mol)

        # Compute the unperturbed HF wfn
        scf0 = magpy.hfwfn(H, self.charge, self.spin, self.print_level)
        scf0.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
        print("Psi4 SCF = ", self.run_psi4_scf(H.molecule))
        ci0 = magpy.ciwfn_so(scf0) # Not strictly necessary, but handy

        # Loop over magnetic field displacements and store wave functions (six total)
        B_pos = []
        B_neg = []
        for B in range(3):
            strength = np.zeros(3)

            # +B displacement
            if self.print_level > 0:
                print("B(%d)+ Displacement" % (B))
            H.reset_V()
            strength[B] = B_disp
            H.add_field(field='magnetic-dipole', strength=strength)
            scf = magpy.hfwfn(H, self.charge, self.spin, self.print_level)
            scf.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
            scf.match_phase(scf0)
            ci = magpy.ciwfn_so(scf)
            ci.solve(e_conv, r_conv, maxiter, max_diis, start_diis)
            B_pos.append(ci)

            # -B displacement
            if self.print_level > 0:
                print("B(%d)- Displacement" % (B))
            H.reset_V()
            strength[B] = -B_disp
            H.add_field(field='magnetic-dipole', strength=strength)
            scf = magpy.hfwfn(H, self.charge, self.spin, self.print_level)
            scf.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
            scf.match_phase(scf0)
            ci = magpy.ciwfn_so(scf)
            ci.solve(e_conv, r_conv, maxiter, max_diis, start_diis)
            B_neg.append(ci)

        # Loop over atomic coordinate displacements
        R_pos = []
        R_neg = []
        for R in range(3*mol.natom()):

            # +R displacement
            if self.print_level > 0:
                print("R(%d)+ Displacement" % (R))
            H = magpy.Hamiltonian(shift_geom(mol, R, R_disp))
            scf = magpy.hfwfn(H, self.charge, self.spin, self.print_level)
            scf.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
            print("Psi4 SCF = ", self.run_psi4_scf(H.molecule))
            scf.match_phase(scf0)
            ci = magpy.ciwfn_so(scf)
            ci.solve(e_conv, r_conv, maxiter, max_diis, start_diis)
            R_pos.append(ci)

            # -R displacement
            if self.print_level > 0:
                print("R(%d)- Displacement" % (R))
            H = magpy.Hamiltonian(shift_geom(mol, R, -R_disp))
            scf = magpy.hfwfn(H, self.charge, self.spin, self.print_level)
            scf.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
            print("Psi4 SCF = ", self.run_psi4_scf(H.molecule))
            scf.match_phase(scf0)
            ci = magpy.ciwfn_so(scf)
            ci.solve(e_conv, r_conv, maxiter, max_diis, start_diis)
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

                S[R][B][0] = self.mo_overlap(R_pos_C, R_pos_H, B_pos_C, B_pos_H)
                S[R][B][1] = self.mo_overlap(R_pos_C, R_pos_H, B_neg_C, B_neg_H)
                S[R][B][2] = self.mo_overlap(R_neg_C, R_neg_H, B_pos_C, B_pos_H)
                S[R][B][3] = self.mo_overlap(R_neg_C, R_neg_H, B_neg_C, B_neg_H)

        # Compute AAT components using finite-difference
        o = slice(0,ci0.no)
        no = ci0.no
        nv = ci0.nv

        # <d0/dR|d0/dB>
        AAT_00 = np.zeros((3*mol.natom(), 3))
        for R in range(3*mol.natom()):
            for B in range(3):

                pp = self.det_overlap([0], [0], S[R][B][0], o)
                pm = self.det_overlap([0], [0], S[R][B][1], o)
                mp = self.det_overlap([0], [0], S[R][B][2], o)
                mm = self.det_overlap([0], [0], S[R][B][3], o)

                # Compute AAT element
                AAT_00[R,B] = (((pp - pm - mp + mm)/(4*R_disp*B_disp))).imag

        AAT_0D = np.zeros((3*mol.natom(), 3))
        AAT_D0 = np.zeros((3*mol.natom(), 3))
        for R in range(3*mol.natom()):
            ci_R_pos = R_pos[R]
            ci_R_neg = R_neg[R]

            for B in range(3):
                ci_B_pos = B_pos[B]
                ci_B_neg = B_neg[B]

                pp = pm = mp = mm = 0.0
                for i in range(no):
                    for a in range(nv):
                        for j in range(no):
                            for b in range(nv):

                                det = self.det_overlap([0], [i, a+no, j, b+no], S[R][B][0], o)
                                pp += 0.25 * ci_B_pos.C2[i,j,a,b] * det
                                det = self.det_overlap([0], [i, a+no, j, b+no], S[R][B][1], o)
                                pm += 0.25 * ci_B_neg.C2[i,j,a,b] * det
                                det = self.det_overlap([0], [i, a+no, j, b+no], S[R][B][2], o)
                                mp += 0.25 * ci_B_pos.C2[i,j,a,b] * det
                                det = self.det_overlap([0], [i, a+no, j, b+no], S[R][B][3], o)
                                mm += 0.25 * ci_B_neg.C2[i,j,a,b] * det

                AAT_0D[R,B] = (((pp - pm - mp + mm)/(4*R_disp*B_disp))).imag

                pp = pm = mp = mm = 0.0
                for i in range(no):
                    for a in range(nv):
                        for j in range(no):
                            for b in range(nv):

                                det = self.det_overlap([i, a+no, j, b+no], [0], S[R][B][0], o)
                                pp += 0.25 * ci_R_pos.C2[i,j,a,b] * det
                                det = self.det_overlap([i, a+no, j, b+no], [0], S[R][B][1], o)
                                pm += 0.25 * ci_R_pos.C2[i,j,a,b] * det
                                det = self.det_overlap([i, a+no, j, b+no], [0], S[R][B][2], o)
                                mp += 0.25 * ci_R_neg.C2[i,j,a,b] * det
                                det = self.det_overlap([i, a+no, j, b+no], [0], S[R][B][3], o)
                                mm += 0.25 * ci_R_neg.C2[i,j,a,b] * det

                AAT_D0[R,B] = (((pp - pm - mp + mm)/(4*R_disp*B_disp))).imag

        AAT_DD = np.zeros((3*mol.natom(), 3))
        s = ["A", "B"]
        for R in range(3*mol.natom()):
            ci_R_pos = R_pos[R]
            ci_R_neg = R_neg[R]

            for B in range(3):
                ci_B_pos = B_pos[B]
                ci_B_neg = B_neg[B]

                pp = pm = mp = mm = 0.0
                print(f"R = {R:2d}; B = {B:2d}")
                for i in range(no):
                    for a in range(nv):
                        for j in range(no):
                            for b in range(nv):
                                for k in range(no):
                                    for c in range(nv):
                                        for l in range(no):
                                            for d in range(nv):

                                                ci_R = ci_R_pos; ci_B = ci_B_pos; disp = 0
                                                det = self.det_overlap([i, a+no, j, b+no], [k, c+no, l, d+no], S[R][B][disp], o)
                                                this = ci_R.C2[i,j,a,b] * ci_B.C2[k,l,c,d] * det
                                                pp += (1/16) * ci_R.C2[i,j,a,b] * ci_B.C2[k,l,c,d] * det

                                                if abs(this.imag) > 1e-12:
                                                    print("%16.13f %1d %1d %1d %1d <%1s%1s%1s%1s | %1s%1s%1s%1s> %1d %1d %1d %1d" % (this.imag, i,a,j,b, s[i%2],s[a%2],s[j%2],s[b%2],s[k%2],s[c%2],s[l%2],s[d%2], k,c,l,d))

                                                ci_R = ci_R_pos; ci_B = ci_B_neg; disp = 1
                                                det = self.det_overlap([i, a+no, j, b+no], [k, c+no, l, d+no], S[R][B][disp], o)
                                                pm += (1/16) * ci_R.C2[i,j,a,b] * ci_B.C2[k,l,c,d] * det

                                                ci_R = ci_R_neg; ci_B = ci_B_pos; disp = 2
                                                det = self.det_overlap([i, a+no, j, b+no], [k, c+no, l, d+no], S[R][B][disp], o)
                                                mp += (1/16) * ci_R.C2[i,j,a,b] * ci_B.C2[k,l,c,d] * det

                                                ci_R = ci_R_neg; ci_B = ci_B_neg; disp = 3
                                                det = self.det_overlap([i, a+no, j, b+no], [k, c+no, l, d+no], S[R][B][disp], o)
                                                mm += (1/16) * ci_R.C2[i,j,a,b] * ci_B.C2[k,l,c,d] * det

                AAT_DD[R,B] = (((pp - pm - mp + mm)/(4*R_disp*B_disp))).imag

        return AAT_00, AAT_0D, AAT_D0, AAT_DD


    def mo_overlap(self, bra, bra_basis, ket, ket_basis):
        """
        Compute the MO overlap matrix between two (possibly different) basis sets

        Parameters
        ----------
        bra: MO coefficient matrix for the bra state (NumPy array)
        bra_basis: Psi4 BasisSet object for the bra state
        ket: MO coefficient matrix for the ket state (NumPy array)
        ket_basis: Psi4 BasisSet object for the ket state

        Returns
        -------
        S: MO-basis overlap matrix (NumPy array)
        """
        # Sanity check
        if (bra.shape[0] != ket.shape[0]) or (bra.shape[1] != ket.shape[1]):
            raise Exception("Bra and Ket States do not have the same dimensions: (%d,%d) vs. (%d,%d)." %
                    (bra.shape[0], bra.shape[1], ket.shape[0], ket.shape[1]))

        # Get AO-basis overlap integrals
        mints = psi4.core.MintsHelper(bra_basis)
        if bra_basis == ket_basis:
            S_ao = mints.ao_overlap().np
        else:
            S_ao = mints.ao_overlap(bra_basis, ket_basis).np

        # Transform to MO basis
        S_mo = bra.T @ S_ao @ ket

        # Convert to spin orbitals
        n = 2 * bra.shape[1]
        S = np.zeros((n,n), dtype=S_mo.dtype)
        for p in range(n):
            for q in range(n):
                S[p,q] = S_mo[p//2,q//2] * (p%2 == q%2)

        return S

    # Compute overlap between two determinants in (possibly) different bases
    def det_overlap(self, bra_indices, ket_indices, S_inp, o):
        """
        Compute the overlap between two Slater determinants (represented by strings of indices)
        of equal length in (possibly) different basis sets using the determinant of their overlap.

        Parameters
        ----------
        bra_indices: list of substitution indices
        ket_indices: list of substitution indices
        S: MO overlap between bra and ket bases (NumPy array)
        o: Slice of S needed for determinant
        """

        S = S_inp.copy()

        if len(bra_indices) == 4: # double excitation
            i = bra_indices[0]; a = bra_indices[1]
            j = bra_indices[2]; b = bra_indices[3]
            S[[a,i],:] = S[[i,a],:]
            S[[b,j],:] = S[[j,b],:]

        if len(ket_indices) == 4: # double excitation
            i = ket_indices[0]; a = ket_indices[1]
            j = ket_indices[2]; b = ket_indices[3]
            S[:,[a,i]] = S[:,[i,a]]
            S[:,[b,j]] = S[:,[j,b]]

        return np.linalg.det(S[o,o])


    def run_psi4_scf(self, molecule):
        geom = molecule.create_psi4_string_from_molecule()
        new_mol = psi4.geometry(geom)
        new_mol.fix_orientation(True)
        new_mol.fix_com(True)
        new_mol.update_geometry()

        return psi4.energy('SCF')

