if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")

import numpy as np
from opt_einsum import contract
import math
import scipy.linalg
from .utils import *

class hfwfn(object):

    def __init__(self, H, charge=0, spin=1, print_level=0):

        # Keep the Hamiltonian (including molecule and basisset)
        self.H = H

        # Determine number of electrons from charge (spin is unused for now)
        nelec = self.nelectron(charge)
        if ((nelec % 2) == 1):
            raise Exception("MagPy is for closed-shell systems only at present.")
        self.ndocc = nelec//2;

        # Determine number of orbitals
        self.nbf = H.basisset.nbf()

        self.print_level = print_level

    def nelectron(self, charge):
        nelec = -charge
        for i in range(self.H.molecule.natom()):
            nelec += self.H.molecule.true_atomic_number(i)

        return nelec

    def solve_scf(self, e_conv=1e-7, r_conv=1e-7, maxiter=100, max_diis=8, start_diis=1):

        print_level = self.print_level

        # Electronic Hamiltonian, including fields
        H = self.H

        # Get the nuclear repulsion energy 
        self.enuc = H.enuc

        # Core Hamiltonian
        h = H.T + H.V

        # Symmetric orthogonalizer
        X = scipy.linalg.fractional_matrix_power(H.S, -0.5)

        # Form the initial guess for density
        F = h.copy()
        Fp = X @ F @ X
        eps, Cp = np.linalg.eigh(Fp)
        C = X @ Cp
        C_occ = C[:,:self.ndocc]
        D = C_occ @ C_occ.T.conj()

        # Compute the initial-guess energy
        escf = contract('ij,ji->', D, (h+F))

        # Setup DIIS object
        diis = DIIS(F, max_diis)

        if print_level > 0:
            print("\n  Nuclear repulsion energy = %20.12f" % self.enuc)
            print("\n Iter     E(elec,real)          E(elec,imag)             E(tot)                Delta(E)              RMS(D)")
            print(" %02d %20.13f %20.13f %20.13f" % (0, escf.real, escf.imag, escf.real + self.enuc))

        # SCF iteration
        for niter in range(1, maxiter+1):

            escf_last = escf
            D_last = D

            # Build the new Fock matrix
            F = h + contract('kl,ijkl->ij', D, (2*H.ERI-H.ERI.swapaxes(1,2)))

            # DIIS extrapolation
            e = (X @ (F @ D @ H.S - (F @ D @ H.S).conj().T) @ X)
            diis.add_error_vector(F, e)
            if niter >= start_diis:
                F = diis.extrapolate(F)

            Fp = X @ F @ X
            eps, Cp = np.linalg.eigh(Fp)
            C = X @ Cp
            C_occ = C[:,:self.ndocc]
            D = C_occ @ C_occ.T.conj()

            escf = contract('ij,ji->', D, (h+F))

            ediff = (escf - escf_last).real
            rms = np.linalg.norm(D-D_last).real

            if print_level > 0:
                print(" %02d %20.13f %20.13f %20.13f %20.13f %20.13f" % (niter, escf.real, escf.imag, escf.real + self.enuc, ediff, rms))

            # Check for convergence
            if ((abs(ediff) < e_conv) and (abs(rms) < r_conv)):
                self.escf = escf
                self.C = C
                self.eps = eps
                if print_level > 0:
                    print("E(SCF) =  %20.15f" % (escf + self.enuc))
                return (escf+self.enuc), C

        # Convergence failure
        raise Exception("SCF iterations failed to converge in %d cycles." % (maxiter))


    def match_phase(self, ref):
        """
        Compute the phases of the MOs in a ket state and match them to those
        of a given bra state

        Parameters
        ----------
        ref: MagPy hfwfn object containing the reference orbitals and basisset

        Returns
        -------
        None, but modifies self.C in place with new phases
        """
        S = mo_overlap(ref.C, ref.H.basisset, self.C, self.H.basisset)

        # Compute normalization constant and phase, and correct phase of ket
        for p in range(self.C.shape[1]):
            N = np.sqrt(S[p][p] * np.conj(S[p][p]))
            phase = S[p][p]/N
            self.C[:, p] *= phase**(-1)
