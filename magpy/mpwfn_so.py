if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")

import numpy as np
import psi4
from opt_einsum import contract
import psi4
from .utils import DIIS


class mpwfn_so(object):

    def __init__(self, hfwfn):

        self.hfwfn = hfwfn

        ## Transform Hamiltonian to MO basis

        # AO-basis one-electron Hamiltonian
        h = self.hfwfn.H.T + self.hfwfn.H.V

        # If there are frozen core orbitals, build and add the frozen-core operator 
        # (core contribution to Fock operator) to the one-electron Hamiltonian
        self.efzc = 0
        nfzc = hfwfn.H.basisset.n_frozen_core()
        if nfzc > 0:
            C = self.hfwfn.C[:,:nfzc] # only core MOs
            Pc = contract('pi,qi->pq', C, C.conj())
            ERI = self.hfwfn.H.ERI
            hc = h + 2.0 * contract('pqrs,pq->rs', ERI, Pc) - contract('pqrs,ps->qr', ERI, Pc)
            self.efzc = contract('pq,pq->', (h+hc), Pc)
            h = hc

        nt = self.nt = 2*(hfwfn.nbf - nfzc) # assumes number of MOs = number of AOs
        no = self.no = 2*(hfwfn.ndocc - nfzc)
        nv = self.nv = self.nt - self.no
        self.nfzc = 2*nfzc

        # Set up orbital subspace slices
        o = self.o = slice(0, no)
        v = self.v = slice(no, nt)
        a = self.a = slice(0, nt)

        # Select active MOs
        C = self.hfwfn.C[:,nfzc:]

        # AO->MO two-electron integral transformation: (ov|ov)
        ERI = self.hfwfn.H.ERI
        ERI = contract('pqrs,sl->pqrl', ERI, C[:,hfwfn.ndocc-nfzc:])
        ERI = contract('pqrl,rk->pqkl', ERI, C.conj()[:,:hfwfn.ndocc-nfzc])
        ERI = contract('pqkl,qj->pjkl', ERI, C[:,hfwfn.ndocc-nfzc:])
        ERI = contract('pjkl,pi->ijkl', ERI, C.conj()[:,:hfwfn.ndocc-nfzc])
        ERI = ERI

        ## Translate Hamiltonian to spin orbital basis
        data = ERI.dtype

        # Convert to Dirac ordering
        ERI_oovv = np.zeros((no, no, nv, nv), dtype=data)
        for p in range(no):
            for q in range(no):
                for r in range(nv):
                    for s in range(nv):
                        ERI_oovv[p,q,r,s] = ERI[p//2,r//2,q//2,s//2] * (p%2 == r%2) * (q%2 == s%2)
                        ERI_oovv[p,q,r,s] -= ERI[p//2,s//2,q//2,r//2] * (p%2 == s%2) * (q%2 == r%2)
        self.ERI_oovv = ERI_oovv

        # AO->MO two-electron integral transformation: (vo|vo)
        ERI = self.hfwfn.H.ERI
        ERI = contract('pqrs,sl->pqrl', ERI, C[:,:hfwfn.ndocc-nfzc])
        ERI = contract('pqrl,rk->pqkl', ERI, C.conj()[:,hfwfn.ndocc-nfzc:])
        ERI = contract('pqkl,qj->pjkl', ERI, C[:,:hfwfn.ndocc-nfzc])
        ERI = contract('pjkl,pi->ijkl', ERI, C.conj()[:,hfwfn.ndocc-nfzc:])

        # Convert to Dirac ordering
        ERI_vvoo = np.zeros((nv, nv, no, no), dtype=data)
        for p in range(nv):
            for q in range(nv):
                for r in range(no):
                    for s in range(no):
                        ERI_vvoo[p,q,r,s] = ERI[p//2,r//2,q//2,s//2] * (p%2 == r%2) * (q%2 == s%2)
                        ERI_vvoo[p,q,r,s] -= ERI[p//2,s//2,q//2,r//2] * (p%2 == s%2) * (q%2 == r%2)
        self.ERI_vvoo = ERI_vvoo

        # Build orbital energy denominators
        eps = hfwfn.eps[nfzc:hfwfn.ndocc]
        eps_occ = np.zeros((no))
        for i in range(no):
            eps_occ[i] = eps[i//2]
        eps = hfwfn.eps[hfwfn.ndocc:]
        eps_vir = np.zeros((nv))
        for a in range(nv):
            eps_vir[a] = eps[a//2]
        Dia = eps_occ.reshape(-1,1) - eps_vir # For later when I add singles
        Dijab = eps_occ.reshape(-1,1,1,1) + eps_occ.reshape(-1,1,1) - eps_vir.reshape(-1,1) - eps_vir
        self.Dijab = Dijab



    def solve(self, **kwargs):

        # Extract kwargs
        print_level = kwargs.pop('print_level', 0)

        valid_normalizations = ['FULL', 'INTERMEDIATE']
        normalization = kwargs.pop('normalization', 'FULL').upper()
        if normalization not in valid_normalizations:
            raise Exception(f"{normalization:s} is not an allowed choice of normalization.")
        self.normalization = normalization

        if print_level > 2:
            print("\nNMO = %d; NACT = %d; NO = %d; NV = %d" % (self.hfwfn.nbf, self.nt, self.no, self.nv))

        o = self.o
        v = self.v
        ERI_vvoo = self.ERI_vvoo
        ERI_oovv = self.ERI_oovv
        Dijab = self.Dijab

        E0 = self.hfwfn.escf + self.hfwfn.H.enuc
        if print_level > 2:
            print("HFWFN ESCF (electronic) = ", self.hfwfn.escf)
            print("HFWFN ESCF (total) =      ", self.hfwfn.escf + self.hfwfn.enuc)

        # first-order wfn amplitudes -- intermediate normalization
        C0 = 1.0
        C2 = ERI_vvoo.swapaxes(0,2).swapaxes(1,3)/Dijab

        # MP2 energy
        emp2 = self.compute_mp2_energy(o, v, ERI_oovv, C2)

        if print_level > 2:
            print("MP2 Correlation Energy = ", emp2)
            print("MP2 Total Energy       = ", emp2 + E0)

        # Re-normalize if necessary
        if self.normalization == 'FULL':
            C0, C2 = self.normalize(o, v, C2)
            norm = np.sqrt(C0*C0 + (1/4) * contract('ijab,ijab', C2.conj(), C2))
            if print_level > 2:
                print(f"Normalization check = {norm:18.12f}")
        self.C0 = C0
        self.C2 = C2

        return emp2, C0, C2

    def compute_mp2_energy(self, o, v, ERI, C2):
        emp2 = (1/4) * contract('ijab,ijab->', C2, ERI)
        return emp2

    def normalize(self, o, v, C2):
        N = 1.0/np.sqrt(1.0 + (1/4) * contract('ijab,ijab->', C2.conj(), C2))
        C0 = N; C2 = N * C2
        return C0, C2
