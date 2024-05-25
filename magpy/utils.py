import psi4
import numpy as np
from itertools import permutations
from opt_einsum import contract
import re
from ast import literal_eval
from multiprocessing import Pool

def levi(indexes):
    """
    Compute the Levi-Civita tensor element for a given list of three indices

    Parameters
    ----------
    indexes: List of three indices (numerical)

    Returns
    -------
    integer: 0 (repeated indices), +1 (even permutation), -1 (odd permutation)

    NB: I found this on a stackoverflow.com post from 3/25/2020
    """
    indexes = list(indexes)
    if len(indexes) != len(set(indexes)):
        return 0
    elif indexes == sorted(indexes):
        return 1
    else:
        for i in range(len(indexes)):
            for j in range(len(indexes) - 1):
                if indexes[i] > indexes[j + 1]:
                    indexes[j], indexes[j + 1] = indexes[j + 1], indexes[j]
                    return -1 * levi(indexes)

def shift_geom(molecule, R, R_disp):
    """
    Shift the R-th Cartesian coordinate of the given molecule geometry by
    R_disp bohr.

    Parameters
    ----------
    molecule: Psi4 Molecule object
    R: R-th Cartesian coordinate, ordered as 0-2 = x, y, z on atom 0, 3-5 =
    x, y, z on atom 1, etc.
    R_disp: displacement size in bohr.

    Returns
    -------
    this_mol: New molecule object with shifted geometry
    """
    # Clone input molecule for this perturbation
    this_mol = molecule.clone()

    # Grab the original geometry and shift the current coordinate
    geom = np.copy(this_mol.geometry().np)
    geom[R//3][R%3] += R_disp
    geom = psi4.core.Matrix.from_array(geom) # Convert to Psi4 Matrix
    this_mol.set_geometry(geom)
    this_mol.fix_orientation(True)
    this_mol.fix_com(True)

    return this_mol

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
    S = bra.T @ S_ao @ ket

    return S


class DIIS(object):
    """
    DIIS solver for SCF and correlated methods.

    """
    def __init__(self, C, max_diis):
        """
        Constructor for DIIS solver.

        Parameters
        ----------
        C: Initial set of coefficients/amplitudes to extrapolate (e.g., Fock matrix, cluster amplitudes, etc.).  The
        coefficients must be provided as a single NumPy array, e.g., different classes of cluster amplitudes (T1, T2,
        etc.) must be concatentated together.

        Returns
        -------
        DIIS object
        """
        self.diis_C = [C.copy()] # List of Fock matrices or concatenated amplitude increment arrays
        self.diis_errors = [] # List of error matrices/vectors
        self.diis_size = 0 # Current DIIS dimension
        self.max_diis = max_diis # Maximum DIIS dimension

    def add_error_vector(self, C, e):
        """
        Add coefficients/amplitudes and error vectors to DIIS space.

        Parameters
        ----------
        C: The coefficients to be extrapolated.  These must be provided as a single NumPy array, e.g., different 
        classes of cluster amplitudes (T1, T2, etc.) must be concatentated together. 
        e: The current error vector.  For SCF methods this should be F @ D @ S - S @ D @ F (possibly in an orthogonal
        basis), and for CI/CC methods the current residuals divided by orbital energy denominators seem to work best.

        Returns
        -------
        None
        """
        self.diis_C.append(C.copy())
        self.diis_errors.append(e.ravel())

    def extrapolate(self, C):
        """
        Extrapolate coefficients.

        Parameters
        ----------
        C: The coefficients to be extrapolated.  These must be provided as a single NumPy array, e.g., different
        classes of cluster amplitudes (T1, T2, etc.) must be concatentated together.

        Returns
        -------
        C: The extrapolated coefficients as a single NumPy array.
        """
        if(self.max_diis == 0):
            return C

        if (len(self.diis_errors) > self.max_diis):
            del self.diis_C[0]
            del self.diis_errors[0]

        self.diis_size = len(self.diis_errors)

        # Build DIIS matrix B
        B = -1 * np.ones((self.diis_size + 1, self.diis_size + 1), dtype=self.diis_errors[0].dtype)
        B[-1, -1] = 0

        for n1, e1 in enumerate(self.diis_errors):
            B[n1, n1] = np.dot(e1.conj(), e1)
            for n2, e2 in enumerate(self.diis_errors):
                if n1 > n2:
                    continue
                B[n1, n2] = np.dot(e1.conj(), e2)
                B[n2, n1] = B[n1, n2]

#        B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

        A = np.zeros((self.diis_size+1))
        A[-1] = -1

        c = np.linalg.solve(B, A)

        C *= 0
        for i in range(self.diis_size):
            C += c[i] * self.diis_C[i+1]

        return C

def make_np_array(a):
    """
    Create a numpy array from the text output of calling print() of a numpy array

    Parameters
    ----------
    a: A printed numpy array, e.g.:
        [[0.53769593 0.89919323 0.4075488  0.36403768]
         [0.72989146 0.2021274  0.97940316 0.68615811]
         [0.90720974 0.13427956 0.4699694  0.92367386]
         [0.31356426 0.75172354 0.78713203 0.45598685]]

       Be sure to put the array in triple quotes (like this documentation) when passing it to the function.

    Returns
    -------
    a: A numpy array.
    """
    a = re.sub(r"([^[])\s+([^]])", r"\1, \2", a)
    a = np.array(literal_eval(a))
    return a

