if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")

import numpy as np
import psi4
from opt_einsum import contract
import psi4
from .utils import DIIS


class ciwfn(object):

    def __init__(self, hfwfn, **kwargs):

        self.hfwfn = hfwfn

        self.nfzc = nfzc = hfwfn.H.basisset.n_frozen_core()

        valid_normalizations = ['FULL', 'INTERMEDIATE']
        normalization = kwargs.pop('normalization', 'FULL').upper()
        if normalization not in valid_normalizations:
            raise Exception(f"{normalization:s} is not an allowed choice of normalization.")
        self.normalization = normalization

        nt = self.nt = hfwfn.nbf - nfzc # assumes number of MOs = number of AOs
        no = self.no = hfwfn.ndocc - nfzc
        nv = self.nv = hfwfn.nbf - self.no - nfzc

        # Set up orbital subspace slices
        o = self.o = slice(0, no)
        v = self.v = slice(no, nt)
        a = self.a = slice(0, nt)

        ## Transform Hamiltonian to MO basis

        # AO-basis one-electron Hamiltonian
        h = self.hfwfn.H.T + self.hfwfn.H.V

        # If there are frozen core orbitals, build and add the frozen-core operator 
        # (core contribution to Fock operator) to the one-electron Hamiltonian
        self.efzc = 0
        if nfzc > 0:
            C = self.hfwfn.C[:,:nfzc] # only core MOs
            Pc = contract('pi,qi->pq', C, C.conj())
            ERI = self.hfwfn.H.ERI
            hc = h + 2.0 * contract('pqrs,pq->rs', ERI, Pc) - contract('pqrs,ps->qr', ERI, Pc)
            self.efzc = contract('pq,pq->', (h+hc), Pc)
            h = hc

        # Select active MOs
        C = self.hfwfn.C[:,nfzc:]

        # AO->MO one-electron integral transformation
        self.h = C.conj().T @ h @ C
        self.h0 = self.h.copy() # Keep original core Hamiltonian

        # AO->MO two-electron integral transformation
        ERI = self.hfwfn.H.ERI
        ERI = contract('pqrs,sl->pqrl', ERI, C)
        ERI = contract('pqrl,rk->pqkl', ERI, C.conj())
        ERI = contract('pqkl,qj->pjkl', ERI, C)
        ERI = contract('pjkl,pi->ijkl', ERI, C.conj())

        # Convert to Dirac ordering and build spin-adapted L
        ERI = self.ERI = ERI.swapaxes(1,2)
        L = self.L = 2.0 * ERI - ERI.swapaxes(2,3)

        # Build MO-basis Fock matrix (diagonal for canonical MOs, but we don't assume them)
        F = self.F = self.h + contract('pmqm->pq', L[a,o,a,o])

        # Build orbital energy denominators
        eps_occ = np.diag(F)[o]
        eps_vir = np.diag(F)[v]
        Dia = eps_occ.reshape(-1,1) - eps_vir # For later when I add singles
        Dijab = eps_occ.reshape(-1,1,1,1) + eps_occ.reshape(-1,1,1) - eps_vir.reshape(-1,1) - eps_vir
        self.Dijab = Dijab


    def solve(self, **kwargs):

        # Extract kwargs
        e_conv = kwargs.pop('e_conv', 1e-7)
        r_conv = kwargs.pop('r_conv', 1e-7)
        maxiter = kwargs.pop('maxiter', 100)
        max_diis = kwargs.pop('max_diis', 8)
        start_diis = kwargs.pop('start_diis', 1)
        print_level = kwargs.pop('print_level', 0)

        if print_level > 2:
            print("\nNMO = %d; NACT = %d; NO = %d; NV = %d" % (self.hfwfn.nbf, self.nt, self.no, self.nv))

        o = self.o
        v = self.v
        no = self.no
        nv = self.nv
        F = self.F
        ERI = self.ERI
        L = self.L
        Dijab = self.Dijab

        # SCF check
        ESCF = self.efzc + 2.0 * contract('ii->',self.h[o,o]) + contract('ijij->', L[o,o,o,o])
        E0 = self.hfwfn.escf + self.hfwfn.H.enuc
        if print_level > 2:
            print("\nESCF (electronic) = ", ESCF)
            print("ESCF (total) =      ", ESCF+self.hfwfn.H.enuc)
            print("HFWFN ESCF (electronic) = ", self.hfwfn.escf)
            print("HFWFN ESCF (total) =      ", self.hfwfn.escf + self.hfwfn.enuc)

        if print_level > 2:
            print("\nSolving projected CID equations.")

        # initial guess amplitudes
        C0 = 1.0
        C2 = ERI[o,o,v,v]/Dijab

        # initial CI energy (= MP2 energy)
        eci = self.compute_cid_energy(o, v, L, C2)

        # Setup DIIS object
        diis = DIIS(C2, max_diis)

        if print_level > 2:
            print("CID Iter %3d: CID Ecorr = %.15f  dE = %.5E  MP2" % (0, eci, -eci))

        ediff = eci
        rmsd = 0.0
        for niter in range(1, maxiter+1):
            eci_last = eci

            r2 = self.r_T2(o, v, eci, F, ERI, L, C2)
            C2 += r2/Dijab
            self.C2 = C2

            rms = contract('ijab,ijab->', r2/Dijab, r2/Dijab)
            rms = np.sqrt(rms)

            eci = self.compute_cid_energy(o, v, L, C2)

            ediff = eci - eci_last

            if print_level > 2:
                print('CID Iter %3d: CID Ecorr = %.15f  dE = %.5E  rms = %.5E' % (niter, eci, ediff, rms))

            if ((abs(ediff) < e_conv) and (abs(rms) < r_conv)):
                if print_level > 2:
                    print("\nCID Equations converged.")
                    print("CID Correlation Energy = ", eci)
                    print("CID Total Energy       = ", eci + E0)

                # Re-normalize if necessary
                if self.normalization == 'FULL':
                    C0, C2 = self.normalize(o, v, C2)
                    norm = np.sqrt(C0*C0 + 2.0 * contract('ijab,ijab->', C2.conj(), C2) - contract('ijab,ijba', C2.conj(), C2))
                    if print_level > 2:
                        print(f"Normalization check = {norm:18.12f}")

                self.C0 = C0
                self.C2 = C2

                return eci, C0, C2

            diis.add_error_vector(C2, r2/Dijab)
            if niter >= start_diis:
                C2 = diis.extrapolate(C2)


    def r_T2(self, o, v, E, F, ERI, L, C2):
        r2 = 0.5 * ERI[v,v,o,o].swapaxes(0,2).swapaxes(1,3).copy()
        r2 += contract('ijae,be->ijab', C2, F[v,v])
        r2 -= contract('imab,mj->ijab', C2, F[o,o])
        r2 += 0.5 * contract('mnab,mnij->ijab', C2, ERI[o,o,o,o])
        r2 += 0.5 * contract('ijef,abef->ijab', C2, ERI[v,v,v,v])

        r2 -= contract('imeb,maje->ijab', C2, ERI[o,v,o,v])
        r2 -= contract('imea,mbej->ijab', C2, ERI[o,v,v,o])
        r2 += contract('miea,mbej->ijab', C2, L[o,v,v,o])

        r2 += r2.swapaxes(0,1).swapaxes(2,3)
        r2 -= E*C2
        return r2


    def compute_cid_energy(self, o, v, L, C2):
        eci = 1.0 * contract('ijab,ijab->', C2, L[o,o,v,v])
        return eci

    def normalize(self, o, v, C2):
        N = 1.0/np.sqrt(1.0 + contract('ijab,ijab->', (2*C2-C2.swapaxes(2,3)).conj(), C2))
        C0 = N; C2 = N * C2
        return C0, C2
