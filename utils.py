import numpy as np
from itertools import permutations
from opt_einsum import contract

def perm_parity(a):
    """
    Compute the parity (+1/-1) of a given permutation of a list of integers 
    from 0,...,N-1.

    Parameters
    ----------
    a: list of integers 0,...,N-1

    Returns
    -------
    integer +1/-1
    """
    parity = 1
    for i in range(0,len(a)-1):
        if a[i] != i:
            parity *= -1
            j = min(range(i,len(a)), key=a.__getitem__)
            a[i],a[j] = a[j],a[i]

    return parity


def mo_overlap(bra, bra_basis, ket, ket_basis):
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
    # Get AO-basis overlap integrals
    mints = psi4.core.MintsHelper(bra_basis)
    if bra_basis = ket_basis:
        S_ao = mints.ao_overlap().np
    else:
        S_ao = mints.ao_overlap(bra_basis, ket_basis).np

    # Transform to MO basis
    S = bra.T @ S_ao @ ket

    return S

def match_phase(bra, bra_basis, ket, ket_basis):
    """
    Compute the phases of the MOs in a ket state and match them to those 
    of a given bra state

    Parameters
    ----------
    bra: MO coefficient matrix for the bra state (NumPy array)
    bra_basis: Psi4 BasisSet object for the bra state
    ket: MO coefficient matrix for the ket state (NumPy array)
    ket_basis: Psi4 BasisSet object for the ket state

    Returns
    -------
    new_ket: The phase-adjusted ket state (NumPy array)
    """
    S = mo_overlap(bra, bra_basis, ket, ket_basis)

    # Compute normalization constant and phase, and correct phase of ket
    new_ket = ket.copy()
    for p in range(nmo):
        N = np.sqrt(S[p][p] * np.conj(S[p][p]))
        phase = S[p][p]/N
        new_ket[:, p] *= phase**(-1)

    return new_ket


# Compute overlap between two determinants in (possibly) different bases
def det_overlap(bra, bra_basis, ket, ket_basis):
    """
    Compute the overlap between two Slater determinants (represented by their
    MO coefficient matrices) of equal length in (possibly) different basis sets

    Parameters
    ----------
    bra: MO coefficient matrix for the bra state (NumPy array)
    bra_basis: Psi4 BasisSet object for the bra state
    ket: MO coefficient matrix for the ket state (NumPy array)
    ket_basis: Psi4 BasisSet object for the ket state

    Returns
    -------
    overlap: scalar
    """
    # Sanity check
    if (bra.shape[0] != ket.shape[0]) or (bra.shape[1] != ket.shape[0]):
        raise Exception("Bra and Ket States do not have the same dimensions: (%d,%d) vs. (%d,%d)." % (bra.shape[0], bra.shape[1], ket.shape[0], ket.shape[1]))

    N = bra.shape[1] # Number of MOs in the determinants
    S = mo_overlap(bra, bra_basis, ket, ket_basis)

    overlap = 0.0
    for perm in permutations(range(N)):
        product = 1.0
        l = list(perm)
        for p in range(N):
            product *= S[p][l[p]] 
        overlap += product*parity(l)

    return overlap

# Compute overlap between two determinants in (possibly) different bases
def det_overlap2(bra, bra_basis, ket, ket_basis):
    """
    Compute the overlap between two Slater determinants (represented by their
    MO coefficient matrices) of equal length in (possibly) different basis
    sets using the determinant of their overlap

    Parameters
    ----------
    bra: MO coefficient matrix for the bra state (NumPy array)
    bra_basis: Psi4 BasisSet object for the bra state
    ket: MO coefficient matrix for the ket state (NumPy array)
    ket_basis: Psi4 BasisSet object for the ket state

    Returns
    -------
    overlap: scalar
    """
    # Sanity check
    if (bra.shape[0] != ket.shape[0]) or (bra.shape[1] != ket.shape[0]):
        raise Exception("Bra and Ket States do not have the same dimensions: (%d,%d) vs. (%d,%d)." % (bra.shape[0], bra.shape[1], ket.shape[0], ket.shape[1]))

    return np.linalg.det(mo_overlap(bra, bra_basis, ket, ket_basis))


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

