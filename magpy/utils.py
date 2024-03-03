import psi4
import numpy as np
from itertools import permutations
from opt_einsum import contract


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
    this_mol.update_geometry()

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
    for p in range(ket.shape[1]):
        N = np.sqrt(S[p][p] * np.conj(S[p][p]))
        phase = S[p][p]/N
        new_ket[:, p] *= phase**(-1)

    return new_ket


# Compute overlap between two determinants in (possibly) different bases
def det_overlap(bra, bra_basis, ket, ket_basis):
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
    if (bra.shape[0] != ket.shape[0]) or (bra.shape[1] != ket.shape[1]):
        raise Exception("Bra and Ket States do not have the same dimensions: (%d,%d) vs. (%d,%d)." % (bra.shape[0], bra.shape[1], ket.shape[0], ket.shape[1]))

    return np.linalg.det(mo_overlap(bra, bra_basis, ket, ket_basis))


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
        B = -1 * np.ones((self.diis_size + 1, self.diis_size + 1))
        B[-1, -1] = 0

        for n1, e1 in enumerate(self.diis_errors):
            B[n1, n1] = np.dot(e1.conjugate(), e1)
            for n2, e2 in enumerate(self.diis_errors):
                if n1 > n2:
                    continue
                B[n1, n2] = np.dot(e1.conjugate(), e2)
                B[n2, n1] = B[n1, n2]

        B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

        A = np.zeros((self.diis_size+1))
        A[-1] = -1

        c = np.linalg.solve(B, A)

        C *= 0
        for i in range(self.diis_size):
            C += c[i] * self.diis_C[i+1]

        return C

