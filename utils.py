import numpy as np
from itertools import permutations
from opt_einsum import contract

def perm_parity(a):
    parity = 1
    for i in range(0,len(a)-1):
        if a[i] != i:
            parity *= -1
            j = min(range(i,len(a)), key=a.__getitem__)
            a[i],a[j] = a[j],a[i]

    return parity


def match_phase(bra, bra_basis, ket, ket_basis):
    # Get AO-basis overlap integrals
    mints = psi4.core.MintsHelper(bra_basis)
    if bra_basis = ket_basis:
        S_ao = mints.ao_overlap().np
    else:
        S_ao = mints.ao_overlap(bra_basis, ket_basis).np

    # Transform to MO basis
    S = bra.T @ S_ao @ ket

    # Compute normalization constant and phase, and correct phase of ket
    new_ket = ket.copy()
    for p in range(nmo):
        N = np.sqrt(S[p][p] * np.conj(S[p][p]))
        phase = S[p][p]/N
        new_ket[:, p] *= phase**(-1)

    return new_ket


def det_overlap(bra, bra_basis, ket, ket_basis):
    # Get AO-basis overlap integrals
    mints = psi4.core.MintsHelper(bra_basis)
    if bra_basis = ket_basis:
        S_ao = mints.ao_overlap().np
    else:
        S_ao = mints.ao_overlap(bra_basis, ket_basis).np

    # Transform to MO basis
    S = bra.T @ S_ao @ ket

    # Compute overlap
    overlap = 0.0
    for perm in permutations(range(ndocc)):
        product = 1.0
        for p in range(ndocc):
            product *= S[p][perm[p]] 
        overlap += product*parity(perm)

    return overlap


class helper_diis(object):
    def __init__(self, F, max_diis):
        self.diis_F = [F.copy()] # List of Fock matrices
        self.diis_errors = [] # List of error matrices
        self.diis_size = 0 # Current DIIS dimension
        self.max_diis = max_diis # Maximum DIIS dimension

    def add_error_vector(self, F, D, S, X):
        self.diis_F.append(F.copy())
        e = X @ (F @ D @ S - (F @ D @ S).conj().T) @ X
        self.diis_errors.append(e)

    def extrapolate(self, F):
        if(self.max_diis == 0):
            return F

        if (len(self.diis_errors) > self.max_diis):
            del self.diis_F[0]
            del self.diis_errors[0]

        self.diis_size = len(self.diis_errors)

        # Build DIIS matrix B
        B = -1 * np.ones((self.diis_size + 1, self.diis_size + 1), dtype=type(F[0,0]))
        B[-1, -1] = 0

        for n1, e1 in enumerate(self.diis_errors):
            B[n1, n1] = contract('pq,pq->', e1.conjugate(), e1)
            for n2, e2 in enumerate(self.diis_errors):
                if n1 > n2:
                    continue
                B[n1, n2] = contract('pq,pq->', e1.conjugate(), e2)
                B[n2, n1] = B[n1, n2]

        A = np.zeros((self.diis_size+1), dtype=type(F[0,0]))
        A[-1] = -1

        c = np.linalg.solve(B, A)

        F[:,:] = 0
        for i in range(self.diis_size):
            F = F + c[i] * self.diis_F[i+1]

        return F

