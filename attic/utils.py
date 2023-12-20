import psi4
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
    if (bra.shape[0] != ket.shape[0]) or (bra.shape[1] != ket.shape[1]):
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
