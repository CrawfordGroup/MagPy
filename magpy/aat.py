if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")

import psi4
import magpy
import numpy as np
from .utils import *
from codetiming import Timer
from multiprocessing import Pool

class AAT(object):

    def __init__(self, molecule, charge=0, spin=1):

        # Ensure geometry remains fixed in space
        molecule.fix_orientation(True)
        molecule.fix_com(True)
        molecule.update_geometry()
        molecule.reinterpret_coordentry(False)
        self.molecule = molecule

        self.charge = charge
        self.spin = spin


    def compute(self, method='HF', R_disp=0.0001, B_disp=0.0001, **kwargs):

        valid_methods = ['HF', 'CID', 'MP2']
        method = method.upper()
        if method not in valid_methods:
            raise Exception(f"{method:s} is not an allowed choice of method.")
        self.method = method

        valid_normalizations = ['FULL', 'INTERMEDIATE']
        normalization = kwargs.pop('normalization', 'FULL').upper()
        if normalization not in valid_normalizations:
            raise Exception(f"{normalization:s} is not an allowed choice of normalization.")
        self.normalization = normalization

        valid_orbitals = ['SPIN', 'SPATIAL']
        orbitals = kwargs.pop('orbitals', 'SPATIAL').upper()
        if orbitals not in valid_orbitals:
            raise Exception(f"{orbitals:s} is not an allowed choice of orbital representation.")
        self.orbitals = orbitals

        # Select parallel algorithm for <D|D> terms
        self.parallel = kwargs.pop('parallel', False)
        if self.parallel is True:
            self.num_procs = kwargs.pop('num_procs', 4)
            print(f"AATs will be computed using parallel algorithm with {self.num_procs:d} processes.")

        # Extract kwargs
        e_conv = kwargs.pop('e_conv', 1e-10)
        r_conv = kwargs.pop('r_conv', 1e-10)
        maxiter = kwargs.pop('maxiter', 400)
        max_diis = kwargs.pop('max_diis', 8)
        start_diis = kwargs.pop('start_diis', 1)
        print_level = kwargs.pop('print_level', 0)

        # Title output
        if print_level >= 1:
            print("\nAtomic Axial Tensor Computation")
            print("=================================")
            print(f"    Method = {method:s}")
            print(f"    Orbitals = {orbitals:s}")
            print(f"    Normalization = {normalization:s}")
            print(f"    parallel = {self.parallel}")
            if self.parallel is True:
                print(f"    num_procs = {self.num_procs:d}")
            print(f"    r_disp = {R_disp:e}")
            print(f"    b_disp = {B_disp:e}")
            print(f"    e_conv = {e_conv:e}")
            print(f"    r_conv = {r_conv:e}")
            print(f"    maxiter = {maxiter:d}")
            print(f"    max_diis = {max_diis:d}")
            print(f"    start_diis = {start_diis:d}")

        mol = self.molecule

        # Compute the unperturbed HF wfn
        H = magpy.Hamiltonian(mol)
        scf0 = magpy.hfwfn(H, self.charge, self.spin)
        scf0.solve(e_conv=e_conv, r_conv=r_conv, maxiter=maxiter, max_diis=max_diis, start_diis=start_diis, print_level=print_level)
        if print_level > 2:
            print("Psi4 SCF = ", self.run_psi4_scf(H.molecule))
        if method == 'CID':
            if orbitals == 'SPATIAL':
                ci0 = magpy.ciwfn(scf0, normalization=normalization)
            else:
                ci0 = magpy.ciwfn_so(scf0, normalization=normalization)
        elif method == 'MP2':
            if orbitals == 'SPATIAL':
                ci0 = magpy.mpwfn(scf0)
            else:
                ci0 = magpy.mpwfn_so(scf0)

        # Magnetic field displacements
        B_pos = []
        B_neg = []
        for B in range(3):
            strength = np.zeros(3)

            # +B displacement
            if print_level > 2:
                print("B(%d)+ Displacement" % (B))
            strength[B] = B_disp
            H = magpy.Hamiltonian(mol)
            H.add_field(field='magnetic-dipole', strength=strength)
            scf = magpy.hfwfn(H, self.charge, self.spin)
            scf.solve(e_conv=e_conv, r_conv=r_conv, maxiter=maxiter, max_diis=max_diis, start_diis=start_diis, print_level=print_level)
            scf.match_phase(scf0)
            if method == 'HF':
                B_pos.append(scf)
            elif method == 'CID':
                if orbitals == 'SPATIAL':
                    ci = magpy.ciwfn(scf, normalization=normalization)
                else:
                    ci = magpy.ciwfn_so(scf, normalization=normalization)
                ci.solve(e_conv=e_conv, r_conv=r_conv, maxiter=maxiter, max_diis=max_diis, start_diis=start_diis, print_level=print_level)
                B_pos.append(ci)
            elif method == 'MP2':
                if orbitals == 'SPATIAL':
                    ci = magpy.mpwfn(scf)
                else:
                    ci = magpy.mpwfn_so(scf)
                ci.solve(normalization=normalization, print_level=print_level)
                B_pos.append(ci)

            # -B displacement
            if print_level > 2:
                print("B(%d)- Displacement" % (B))
            strength[B] = -B_disp
            H = magpy.Hamiltonian(mol)
            H.add_field(field='magnetic-dipole', strength=strength)
            scf = magpy.hfwfn(H, self.charge, self.spin)
            scf.solve(e_conv=e_conv, r_conv=r_conv, maxiter=maxiter, max_diis=max_diis, start_diis=start_diis, print_level=print_level)
            scf.match_phase(scf0)
            if method == 'HF':
                B_neg.append(scf)
            elif method == 'CID':
                if orbitals == 'SPATIAL':
                    ci = magpy.ciwfn(scf, normalization=normalization)
                else:
                    ci = magpy.ciwfn_so(scf, normalization=normalization)
                ci.solve(e_conv=e_conv, r_conv=r_conv, maxiter=maxiter, max_diis=max_diis, start_diis=start_diis, print_level=print_level)
                B_neg.append(ci)
            elif method == 'MP2':
                if orbitals == 'SPATIAL':
                    ci = magpy.mpwfn(scf)
                else:
                    ci = magpy.mpwfn_so(scf)
                ci.solve(normalization=normalization, print_level=print_level)
                B_neg.append(ci)

        # Atomic coordinate displacements
        R_pos = []
        R_neg = []
        for R in range(3*mol.natom()):

            # +R displacement
            if print_level > 2:
                print("R(%d)+ Displacement" % (R))
            H = magpy.Hamiltonian(shift_geom(mol, R, R_disp))
            rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
            scf = magpy.hfwfn(H, self.charge, self.spin)
            scf.solve(e_conv=e_conv, r_conv=r_conv, maxiter=maxiter, max_diis=max_diis, start_diis=start_diis, print_level=print_level)
            if print_level > 2:
                print("Psi4 SCF = ", self.run_psi4_scf(H.molecule))
            scf.match_phase(scf0)
            if method == 'HF':
                R_pos.append(scf)
            elif method == 'CID':
                if orbitals == 'SPATIAL':
                    ci = magpy.ciwfn(scf, normalization=normalization)
                else:
                    ci = magpy.ciwfn_so(scf, normalization=normalization)
                ci.solve(e_conv=e_conv, r_conv=r_conv, maxiter=maxiter, max_diis=max_diis, start_diis=start_diis, print_level=print_level)
                R_pos.append(ci)
            elif method == 'MP2':
                if orbitals == 'SPATIAL':
                    ci = magpy.mpwfn(scf)
                else:
                    ci = magpy.mpwfn_so(scf)
                ci.solve(normalization=normalization, print_level=print_level)
                R_pos.append(ci)

            # -R displacement
            if print_level > 2:
                print("R(%d)- Displacement" % (R))
            H = magpy.Hamiltonian(shift_geom(mol, R, -R_disp))
            scf = magpy.hfwfn(H, self.charge, self.spin)
            scf.solve(e_conv=e_conv, r_conv=r_conv, maxiter=maxiter, max_diis=max_diis, start_diis=start_diis, print_level=print_level)
            if print_level > 2:
                print("Psi4 SCF = ", self.run_psi4_scf(H.molecule))
            scf.match_phase(scf0)
            if method == 'HF':
                R_neg.append(scf)
            elif method == 'CID':
                if orbitals == 'SPATIAL':
                    ci = magpy.ciwfn(scf, normalization=normalization)
                else:
                    ci = magpy.ciwfn_so(scf, normalization=normalization)
                ci.solve(e_conv=e_conv, r_conv=r_conv, maxiter=maxiter, max_diis=max_diis, start_diis=start_diis, print_level=print_level)
                R_neg.append(ci)
            elif method == 'MP2':
                if orbitals == 'SPATIAL':
                    ci = magpy.mpwfn(scf)
                else:
                    ci = magpy.mpwfn_so(scf)
                ci.solve(normalization=normalization, print_level=print_level)
                R_neg.append(ci)

        # Compute full MO overlap matrix for all combinations of perturbed MOs
        S = [[[0 for k in range(4)] for j in range(3)] for i in range(3*mol.natom())] # list of overlap matrices
        for R in range(3*mol.natom()):
            if method == 'HF':
                R_pos_C = R_pos[R].C
                R_neg_C = R_neg[R].C
                R_pos_H = R_pos[R].H.basisset
                R_neg_H = R_neg[R].H.basisset
            else:
                R_pos_C = R_pos[R].hfwfn.C
                R_neg_C = R_neg[R].hfwfn.C
                R_pos_H = R_pos[R].hfwfn.H.basisset
                R_neg_H = R_neg[R].hfwfn.H.basisset

            for B in range(3):
                if method == 'HF':
                    B_pos_C = B_pos[B].C
                    B_neg_C = B_neg[B].C
                    B_pos_H = B_pos[B].H.basisset
                    B_neg_H = B_neg[B].H.basisset
                else:
                    B_pos_C = B_pos[B].hfwfn.C
                    B_neg_C = B_neg[B].hfwfn.C
                    B_pos_H = B_pos[B].hfwfn.H.basisset
                    B_neg_H = B_neg[B].hfwfn.H.basisset

                S[R][B][0] = self.mo_overlap(R_pos_C, R_pos_H, B_pos_C, B_pos_H)
                S[R][B][1] = self.mo_overlap(R_pos_C, R_pos_H, B_neg_C, B_neg_H)
                S[R][B][2] = self.mo_overlap(R_neg_C, R_neg_H, B_pos_C, B_pos_H)
                S[R][B][3] = self.mo_overlap(R_neg_C, R_neg_H, B_neg_C, B_neg_H)

        # Compute AAT components using finite-difference
        if method == 'HF':
            o = slice(0,scf0.ndocc)
        elif method == 'CID' or method == 'MP2':
            o = slice(0,ci0.no+ci0.nfzc) # Used only for the dimension of the sub-matrices of which we're taking the determinants
            no = ci0.no
            nv = ci0.nv
            nfzc = ci0.nfzc

        # <d0/dR|d0/dB>
        AAT_00 = np.zeros((3*mol.natom(), 3))
        for R in range(3*mol.natom()):
            if method == 'HF':
                C0_R_pos = C0_R_neg = 1.0
            else:
                C0_R_pos = R_pos[R].C0
                C0_R_neg = R_neg[R].C0

            for B in range(3):
                if method == 'HF':
                    C0_B_pos = C0_B_neg = 1.0
                else:
                    C0_B_pos = B_pos[B].C0
                    C0_B_neg = B_neg[B].C0

                if method == 'HF':
                    pp = np.linalg.det(S[R][B][0][o,o])
                    pm = np.linalg.det(S[R][B][1][o,o])
                    mp = np.linalg.det(S[R][B][2][o,o])
                    mm = np.linalg.det(S[R][B][3][o,o])
                    AAT_00[R,B] = 2*(((pp - pm - mp + mm)/(4*R_disp*B_disp))).imag
                else:
                    pp = det_overlap(self.orbitals, [0], [0], S[R][B][0], o, spins='AAAA') * C0_R_pos * C0_B_pos
                    pm = det_overlap(self.orbitals, [0], [0], S[R][B][1], o, spins='AAAA') * C0_R_pos * C0_B_neg
                    mp = det_overlap(self.orbitals, [0], [0], S[R][B][2], o, spins='AAAA') * C0_R_neg * C0_B_pos
                    mm = det_overlap(self.orbitals, [0], [0], S[R][B][3], o, spins='AAAA') * C0_R_neg * C0_B_neg
                    AAT_00[R,B] = (((pp - pm - mp + mm)/(4*R_disp*B_disp))).imag

        if print_level >= 1:
            print(f"Hartree-Fock AAT (normalization = {self.normalization:s}):")
            print(AAT_00)

        if method == 'HF':
            return AAT_00

        AAT_0D = np.zeros((3*mol.natom(), 3))
        AAT_D0 = np.zeros((3*mol.natom(), 3))
        for R in range(3*mol.natom()):
            ci_R_pos = R_pos[R]
            ci_R_neg = R_neg[R]

            for B in range(3):
                ci_B_pos = B_pos[B]
                ci_B_neg = B_neg[B]

                # <d0/dR|dD/dB>
                pp, pm, mp, mm = self.AAT_0D(ci_R_pos, ci_R_neg, ci_B_pos, ci_B_neg, S[R][B], o)
                AAT_0D[R,B] = (((pp - pm - mp + mm)/(4*R_disp*B_disp))).imag

                # <dD/dR|d0/dB>
                pp, pm, mp, mm = self.AAT_D0(ci_R_pos, ci_R_neg, ci_B_pos, ci_B_neg, S[R][B], o)
                AAT_D0[R,B] = (((pp - pm - mp + mm)/(4*R_disp*B_disp))).imag

        orbitals = self.orbitals
        if self.parallel is True:
            pool = Pool(processes=self.num_procs)

            args = [] # argument list for each R/B combination
            for R in range(3*mol.natom()):
                for B in range(3):
                    args.append([R_disp, B_disp, R_pos[R].C2, R_neg[R].C2, B_pos[B].C2, B_neg[B].C2, S[R][B], orbitals, nfzc])

            result = pool.starmap_async(AAT_DD_element, args)
            AAT_DD = np.asarray(result.get()).reshape(3*mol.natom(), 3)
        else:
            AAT_DD = np.zeros((3*mol.natom(), 3))
            for R in range(3*mol.natom()):
                for B in range(3):
                    print(f"Atom = {R//3:d}; Coord = {R%3:d}; Field = {B:d}")

                    # <dD/dR|dD/dB>
                    AAT_DD[R,B] = AAT_DD_element(R_disp, B_disp, R_pos[R].C2, R_neg[R].C2, B_pos[B].C2, B_neg[B].C2, S[R][B], orbitals)

        if print_level >= 1:
            print("Correlated AAT (normalization = {self.normalization}):")
            print(AAT_DD)
            print("Total electronic AAT (normalization = {self.normalization}):")
            print(AAT_00 + AAT_DD)

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
        if self.orbitals == 'SPIN':
            n = 2 * bra.shape[1]
            S = np.zeros((n,n), dtype=S_mo.dtype)
            for p in range(n):
                for q in range(n):
                    S[p,q] = S_mo[p//2,q//2] * (p%2 == q%2)
            return S
        else:
             return S_mo


    def AAT_0D(self, ci_R_pos, ci_R_neg, ci_B_pos, ci_B_neg, S, o):
        no = ci_R_pos.no
        nv = ci_R_pos.nv
        nfzc = ci_R_pos.nfzc

        pp = pm = mp = mm = 0.0
        if self.orbitals == 'SPATIAL':
            for i in range(no):
                I = i + nfzc
                for a in range(nv):
                    A = a + no + nfzc
                    for j in range(no):
                        J = j + nfzc
                        for b in range(nv):
                            B = b + no + nfzc
                            det_AA = det_overlap(self.orbitals, [0], [I, A, J, B], S[0], o, spins='AAAA')
                            det_AB = det_overlap(self.orbitals, [0], [I, A, J, B], S[0], o, spins='ABAB')
                            pp += (0.5 * (ci_B_pos.C2[i,j,a,b] - ci_B_pos.C2[i,j,b,a]) * det_AA + ci_B_pos.C2[i,j,a,b] * det_AB) * ci_R_pos.C0

                            det_AA = det_overlap(self.orbitals, [0], [I, A, J, B], S[1], o, spins='AAAA')
                            det_AB = det_overlap(self.orbitals, [0], [I, A, J, B], S[1], o, spins='ABAB')
                            pm += (0.5 * (ci_B_neg.C2[i,j,a,b] - ci_B_neg.C2[i,j,b,a]) * det_AA + ci_B_neg.C2[i,j,a,b] * det_AB) * ci_R_pos.C0

                            det_AA = det_overlap(self.orbitals, [0], [I, A, J, B], S[2], o, spins='AAAA')
                            det_AB = det_overlap(self.orbitals, [0], [I, A, J, B], S[2], o, spins='ABAB')
                            mp += (0.5 * (ci_B_pos.C2[i,j,a,b] - ci_B_pos.C2[i,j,b,a]) * det_AA + ci_B_pos.C2[i,j,a,b] * det_AB) * ci_R_neg.C0

                            det_AA = det_overlap(self.orbitals, [0], [I, A, J, B], S[3], o, spins='AAAA')
                            det_AB = det_overlap(self.orbitals, [0], [I, A, J, B], S[3], o, spins='ABAB')
                            mm += (0.5 * (ci_B_neg.C2[i,j,a,b] - ci_B_neg.C2[i,j,b,a]) * det_AA + ci_B_neg.C2[i,j,a,b] * det_AB) * ci_R_neg.C0

        elif self.orbitals == 'SPIN':
            for i in range(0, no, 1):
                I = i + nfzc
                for a in range(0, nv, 1):
                    A = a + no + nfzc
                    for j in range(0, no, 1):
                        J = j + nfzc
                        for b in range(0, nv, 1):
                            B = b + no + nfzc
                            det = det_overlap(self.orbitals, [0], [I, A, J, B], S[0], o)
                            pp += 0.25 * ci_B_pos.C2[i,j,a,b] * det * ci_R_pos.C0
                            det = det_overlap(self.orbitals, [0], [I, A, J, B], S[1], o)
                            pm += 0.25 * ci_B_neg.C2[i,j,a,b] * det * ci_R_pos.C0
                            det = det_overlap(self.orbitals, [0], [I, A, J, B], S[2], o)
                            mp += 0.25 * ci_B_pos.C2[i,j,a,b] * det * ci_R_neg.C0
                            det = det_overlap(self.orbitals, [0], [I, A, J, B], S[3], o)
                            mm += 0.25 * ci_B_neg.C2[i,j,a,b] * det * ci_R_neg.C0

        return pp, pm, mp, mm

    def AAT_D0(self, ci_R_pos, ci_R_neg, ci_B_pos, ci_B_neg, S, o):
        no = ci_R_pos.no
        nv = ci_R_pos.nv
        nfzc = ci_R_pos.nfzc

        pp = pm = mp = mm = 0.0
        if self.orbitals == 'SPATIAL':
            for i in range(no):
                I = i + nfzc
                for a in range(nv):
                    A = a + no + nfzc
                    for j in range(no):
                        J = j + nfzc
                        for b in range(nv):
                            B = b + no + nfzc
                            det_AA = det_overlap(self.orbitals, [I, A, J, B], [0], S[0], o, spins='AAAA')
                            det_AB = det_overlap(self.orbitals, [I, A, J, B], [0], S[0], o, spins='ABAB')
                            pp += (0.5 * (ci_R_pos.C2[i,j,a,b] - ci_R_pos.C2[i,j,b,a]) * det_AA + ci_R_pos.C2[i,j,a,b] * det_AB) * ci_B_pos.C0

                            det_AA = det_overlap(self.orbitals, [I, A, J, B], [0], S[1], o, spins='AAAA')
                            det_AB = det_overlap(self.orbitals, [I, A, J, B], [0], S[1], o, spins='ABAB')
                            pm += (0.5 * (ci_R_pos.C2[i,j,a,b] - ci_R_pos.C2[i,j,b,a]) * det_AA + ci_R_pos.C2[i,j,a,b] * det_AB) * ci_B_neg.C0

                            det_AA = det_overlap(self.orbitals, [I, A, J, B], [0], S[2], o, spins='AAAA')
                            det_AB = det_overlap(self.orbitals, [I, A, J, B], [0], S[2], o, spins='ABAB')
                            mp += (0.5 * (ci_R_neg.C2[i,j,a,b] - ci_R_neg.C2[i,j,b,a]) * det_AA + ci_R_neg.C2[i,j,a,b] * det_AB) * ci_B_pos.C0

                            det_AA = det_overlap(self.orbitals, [I, A, J, B], [0], S[3], o, spins='AAAA')
                            det_AB = det_overlap(self.orbitals, [I, A, J, B], [0], S[3], o, spins='ABAB')
                            mm += (0.5 * (ci_R_neg.C2[i,j,a,b] - ci_R_neg.C2[i,j,b,a]) * det_AA + ci_R_neg.C2[i,j,a,b] * det_AB) * ci_B_neg.C0

        elif self.orbitals == 'SPIN':
            for i in range(0, no, 1):
                I = i + nfzc
                for a in range(0, nv, 1):
                    A = a + no + nfzc
                    for j in range(0, no, 1):
                        J = j + nfzc
                        for b in range(0, nv, 1):
                            B = b + no + nfzc
                            det = det_overlap(self.orbitals, [I, A, J, B], [0], S[0], o)
                            pp += 0.25 * ci_R_pos.C2[i,j,a,b] * det * ci_B_pos.C0
                            det = det_overlap(self.orbitals, [I, A, J, B], [0], S[1], o)
                            pm += 0.25 * ci_R_pos.C2[i,j,a,b] * det * ci_B_neg.C0
                            det = det_overlap(self.orbitals, [I, A, J, B], [0], S[2], o)
                            mp += 0.25 * ci_R_neg.C2[i,j,a,b] * det * ci_B_pos.C0
                            det = det_overlap(self.orbitals, [I, A, J, B], [0], S[3], o)
                            mm += 0.25 * ci_R_neg.C2[i,j,a,b] * det * ci_B_neg.C0

        return pp, pm, mp, mm

    def nuclear(self):
        """
        Computes the nuclear contribution to the atomic axial tensor (AAT).

        Parameters
        ----------
        None

        Returns
        -------
        aat_nuc: N*3 x 3 array of nuclear contributions to AAT
        """

        geom, mass, elem, Z, uniq = self.molecule.to_arrays()
        natom = self.molecule.natom()

        AAT = np.zeros((natom*3,3))
        for M in range(natom):
            for alpha in range(3): # atomic Cartesian coordinate
                R = M*3 + alpha
                for beta in range(3): # magnetic field coordinate
                    val = 0.0
                    for gamma in range(3): # atomic Cartesian coordinate
                        AAT[R,beta] += (1/4) * levi([alpha, beta, gamma]) * geom[M, gamma] * Z[M]

        return AAT


    def run_psi4_scf(self, molecule):
        geom = molecule.create_psi4_string_from_molecule()
        new_mol = psi4.geometry(geom)
        new_mol.fix_orientation(True)
        new_mol.fix_com(True)
        new_mol.update_geometry()

        return psi4.energy('SCF')


@Timer()
def AAT_DD_element(R_disp, B_disp, C2_R_pos, C2_R_neg, C2_B_pos, C2_B_neg, S, orbitals, nfzc):
    no = C2_R_pos.shape[0]
    nv = C2_R_pos.shape[2]
    o = slice(0,no+nfzc)

    pp = pm = mp = mm = 0.0
    if orbitals == 'SPATIAL':
        for i in range(no):
            I = i + nfzc
            for a in range(nv):
                A = a + no + nfzc
                for j in range(no):
                    J = j + nfzc
                    for b in range(nv):
                        B = b + no + nfzc
                        for k in range(no):
                            K = k + nfzc
                            for c in range(nv):
                                C = c + no + nfzc
                                for l in range(no):
                                    L = l + nfzc
                                    for d in range(nv):
                                        D = d + no + nfzc
                                        C2_R = C2_R_pos; C2_B = C2_B_pos; disp = 0
                                        det_AA_AA = det_overlap(orbitals, [I, A, J, B], [K, C, L, D], S[disp], o, spins='AAAA')
                                        det_AA_BB = det_overlap(orbitals, [I, A, J, B], [K, C, L, D], S[disp], o, spins='AABB')
                                        det_AB_AB = det_overlap(orbitals, [I, A, J, B], [K, C, L, D], S[disp], o, spins='ABAB')
                                        det_AA_AB = det_overlap(orbitals, [I, A, J, B], [K, C, L, D], S[disp], o, spins='AAAB')
                                        det_AB_AA = det_overlap(orbitals, [I, A, J, B], [K, C, L, D], S[disp], o, spins='ABAA')
                                        pp += (1/8) * (C2_R[i,j,a,b] - C2_R[i,j,b,a]) * (C2_B[k,l,c,d] - C2_B[k,l,d,c]) *det_AA_AA
                                        pp += (1/8) * (C2_R[i,j,a,b] - C2_R[i,j,b,a]) * (C2_B[k,l,c,d] - C2_B[k,l,d,c]) *det_AA_BB
                                        pp += (1/2) * (C2_R[i,j,a,b] - C2_R[i,j,b,a]) * C2_B[k,l,c,d] *det_AA_AB
                                        pp += (1/2) * C2_R[i,j,a,b] * (C2_B[k,l,c,d] - C2_B[k,l,d,c]) *det_AB_AA
                                        pp += C2_R[i,j,a,b] * C2_B[k,l,c,d] *det_AB_AB

                                        C2_R = C2_R_pos; C2_B = C2_B_neg; disp = 1
                                        det_AA_AA = det_overlap(orbitals, [I, A, J, B], [K, C, L, D], S[disp], o, spins='AAAA')
                                        det_AA_BB = det_overlap(orbitals, [I, A, J, B], [K, C, L, D], S[disp], o, spins='AABB')
                                        det_AB_AB = det_overlap(orbitals, [I, A, J, B], [K, C, L, D], S[disp], o, spins='ABAB')
                                        det_AA_AB = det_overlap(orbitals, [I, A, J, B], [K, C, L, D], S[disp], o, spins='AAAB')
                                        det_AB_AA = det_overlap(orbitals, [I, A, J, B], [K, C, L, D], S[disp], o, spins='ABAA')
                                        pm += (1/8) * (C2_R[i,j,a,b] - C2_R[i,j,b,a]) * (C2_B[k,l,c,d] - C2_B[k,l,d,c]) *det_AA_AA
                                        pm += (1/8) * (C2_R[i,j,a,b] - C2_R[i,j,b,a]) * (C2_B[k,l,c,d] - C2_B[k,l,d,c]) *det_AA_BB
                                        pm += (1/2) * (C2_R[i,j,a,b] - C2_R[i,j,b,a]) * C2_B[k,l,c,d] *det_AA_AB
                                        pm += (1/2) * C2_R[i,j,a,b] * (C2_B[k,l,c,d] - C2_B[k,l,d,c]) *det_AB_AA
                                        pm += C2_R[i,j,a,b] * C2_B[k,l,c,d] *det_AB_AB

                                        C2_R = C2_R_neg; C2_B = C2_B_pos; disp = 2
                                        det_AA_AA = det_overlap(orbitals, [I, A, J, B], [K, C, L, D], S[disp], o, spins='AAAA')
                                        det_AA_BB = det_overlap(orbitals, [I, A, J, B], [K, C, L, D], S[disp], o, spins='AABB')
                                        det_AB_AB = det_overlap(orbitals, [I, A, J, B], [K, C, L, D], S[disp], o, spins='ABAB')
                                        det_AA_AB = det_overlap(orbitals, [I, A, J, B], [K, C, L, D], S[disp], o, spins='AAAB')
                                        det_AB_AA = det_overlap(orbitals, [I, A, J, B], [K, C, L, D], S[disp], o, spins='ABAA')
                                        mp += (1/8) * (C2_R[i,j,a,b] - C2_R[i,j,b,a]) * (C2_B[k,l,c,d] - C2_B[k,l,d,c]) *det_AA_AA
                                        mp += (1/8) * (C2_R[i,j,a,b] - C2_R[i,j,b,a]) * (C2_B[k,l,c,d] - C2_B[k,l,d,c]) *det_AA_BB
                                        mp += (1/2) * (C2_R[i,j,a,b] - C2_R[i,j,b,a]) * C2_B[k,l,c,d] *det_AA_AB
                                        mp += (1/2) * C2_R[i,j,a,b] * (C2_B[k,l,c,d] - C2_B[k,l,d,c]) *det_AB_AA
                                        mp += C2_R[i,j,a,b] * C2_B[k,l,c,d] *det_AB_AB

                                        C2_R = C2_R_neg; C2_B = C2_B_neg; disp = 3
                                        det_AA_AA = det_overlap(orbitals, [I, A, J, B], [K, C, L, D], S[disp], o, spins='AAAA')
                                        det_AA_BB = det_overlap(orbitals, [I, A, J, B], [K, C, L, D], S[disp], o, spins='AABB')
                                        det_AB_AB = det_overlap(orbitals, [I, A, J, B], [K, C, L, D], S[disp], o, spins='ABAB')
                                        det_AA_AB = det_overlap(orbitals, [I, A, J, B], [K, C, L, D], S[disp], o, spins='AAAB')
                                        det_AB_AA = det_overlap(orbitals, [I, A, J, B], [K, C, L, D], S[disp], o, spins='ABAA')
                                        mm += (1/8) * (C2_R[i,j,a,b] - C2_R[i,j,b,a]) * (C2_B[k,l,c,d] - C2_B[k,l,d,c]) *det_AA_AA
                                        mm += (1/8) * (C2_R[i,j,a,b] - C2_R[i,j,b,a]) * (C2_B[k,l,c,d] - C2_B[k,l,d,c]) *det_AA_BB
                                        mm += (1/2) * (C2_R[i,j,a,b] - C2_R[i,j,b,a]) * C2_B[k,l,c,d] *det_AA_AB
                                        mm += (1/2) * C2_R[i,j,a,b] * (C2_B[k,l,c,d] - C2_B[k,l,d,c]) *det_AB_AA
                                        mm += C2_R[i,j,a,b] * C2_B[k,l,c,d] *det_AB_AB

    elif orbitals == 'SPIN':
        for i in range(0, no, 1):
            I = i + nfzc
            for a in range(0, nv, 1):
                A = a + no + nfzc
                for j in range(0, no, 1):
                    J = j + nfzc
                    for b in range(0, nv, 1):
                        B = b + no + nfzc
                        for k in range(0, no, 1):
                            K = k + nfzc
                            for c in range(0, nv, 1):
                                C = c + no + nfzc
                                for l in range(0, no, 1):
                                    L = l + nfzc
                                    for d in range(0, nv, 1):
                                        D = d + no + nfzc
                                        C2_R = C2_R_pos; C2_B = C2_B_pos; disp = 0
                                        det = det_overlap(orbitals, [I, A, J, B], [K, C, L, D], S[disp], o)
                                        pp += (1/16) * C2_R[i,j,a,b] * C2_B[k,l,c,d] * det

                                        C2_R = C2_R_pos; C2_B = C2_B_neg; disp = 1
                                        det = det_overlap(orbitals, [I, A, J, B], [K, C, L, D], S[disp], o)
                                        pm += (1/16) * C2_R[i,j,a,b] * C2_B[k,l,c,d] * det

                                        C2_R = C2_R_neg; C2_B = C2_B_pos; disp = 2
                                        det = det_overlap(orbitals, [I, A, J, B], [K, C, L, D], S[disp], o)
                                        mp += (1/16) * C2_R[i,j,a,b] * C2_B[k,l,c,d] * det

                                        C2_R = C2_R_neg; C2_B = C2_B_neg; disp = 3
                                        det = det_overlap(orbitals, [I, A, J, B], [K, C, L, D], S[disp], o)
                                        mm += (1/16) * C2_R[i,j,a,b] * C2_B[k,l,c,d] * det

    return (((pp - pm - mp + mm)/(4*R_disp*B_disp))).imag


# Compute overlap between two determinants in (possibly) different bases
def det_overlap(orbitals, bra_indices, ket_indices, S, o, spins='AAAA'):
    """
    Compute the overlap between two Slater determinants (represented by strings of indices)
    of equal length in (possibly) different basis sets using the determinant of their overlap.

    Parameters
    ----------
    bra_indices: list of substitution indices
    ket_indices: list of substitution indices
    S: MO overlap between bra and ket bases (NumPy array)
    o: Slice of S needed for determinant
    spins: 'AAAA', 'AAAB', 'ABAA', or 'ABAB' (string)
    """

    if orbitals == 'SPIN':
        S = S.copy()

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

    elif orbitals == 'SPATIAL':
        S_alpha = S.copy()
        S_beta = S.copy()

        if len(spins) != 4:
            raise Exception(f"Excitations currently limited to doubles only: {spins:s}")

        bra_spin = spins[0] + spins[1]
        ket_spin = spins[2] + spins[3]

        if len(bra_indices) == 4: # double excitation
            i = bra_indices[0]; a = bra_indices[1]
            j = bra_indices[2]; b = bra_indices[3]
            if bra_spin == 'AA':
                S_alpha[[a,i],:] = S_alpha[[i,a],:]
                S_alpha[[b,j],:] = S_alpha[[j,b],:]
            elif bra_spin == 'AB':
                S_alpha[[a,i],:] = S_alpha[[i,a],:]
                S_beta[[b,j],:] = S_beta[[j,b],:]
            elif bra_spin == 'BB':
                S_beta[[a,i],:] = S_beta[[i,a],:]
                beta[[b,j],:] = S_beta[[j,b],:]
        if len(ket_indices) == 4: # double excitation
            i = ket_indices[0]; a = ket_indices[1]
            j = ket_indices[2]; b = ket_indices[3]
            if ket_spin == 'AA':
                S_alpha[:,[a,i]] = S_alpha[:,[i,a]]
                S_alpha[:,[b,j]] = S_alpha[:,[j,b]]
            elif ket_spin == 'AB':
                S_alpha[:,[a,i]] = S_alpha[:,[i,a]]
                S_beta[:,[b,j]] = S_beta[:,[j,b]]
            elif bra_spin == 'BB':
                S_beta[[a,i],:] = S_beta[[i,a],:]
                beta[[b,j],:] = S_beta[[j,b],:] 

        return np.linalg.det(S_alpha[o,o])*np.linalg.det(S_beta[o,o]) 
    else:               
        raise Exception("{orbitals:s} is not an allowed choice of orbital representation.") 
    
