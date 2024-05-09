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

        B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

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


def AAT_DD_parallel(procs, natom, R_disp, B_disp, R_pos_amps, R_neg_amps, B_pos_amps, B_neg_amps, S, orbitals):

    pool = Pool(processes=procs)

    args = [] # argument list for each R/B combination
    for R in range(3*natom):
        for B in range(3):
            args.append([R_disp, B_disp, R_pos_amps[R], R_neg_amps[R], B_pos_amps[B], B_neg_amps[B], S[R][B], orbitals])

    result = pool.starmap_async(AAT_DD_parallel_element, args)
    print(result.get())
    return np.asarray(result.get()).reshape(3*natom,3)


def AAT_DD_parallel_element(R_disp, B_disp, C2_R_pos, C2_R_neg, C2_B_pos, C2_B_neg, S, orbitals):
    no = C2_R_pos.shape[0]
    nv = C2_R_pos.shape[2]
    o = slice(0,no)

    pp = pm = mp = mm = 0.0
    if orbitals == 'SPATIAL':
        for i in range(no):
            for a in range(nv):
                for j in range(no):
                    for b in range(nv):
                        for k in range(no):
                            for c in range(nv):
                                for l in range(no):
                                    for d in range(nv):
                                        C2_R = C2_R_pos; C2_B = C2_B_pos; disp = 0
                                        det_AA_AA = det_overlap(orbitals, [i, a+no, j, b+no], [k, c+no, l, d+no], S[disp], o, spins='AAAA')
                                        det_AA_BB = det_overlap(orbitals, [i, a+no, j, b+no], [k, c+no, l, d+no], S[disp], o, spins='AABB')
                                        det_AB_AB = det_overlap(orbitals, [i, a+no, j, b+no], [k, c+no, l, d+no], S[disp], o, spins='ABAB')
                                        det_AA_AB = det_overlap(orbitals, [i, a+no, j, b+no], [k, c+no, l, d+no], S[disp], o, spins='AAAB')
                                        det_AB_AA = det_overlap(orbitals, [i, a+no, j, b+no], [k, c+no, l, d+no], S[disp], o, spins='ABAA')
                                        pp += (1/8) * (C2_R[i,j,a,b] - C2_R[i,j,b,a]) * (C2_B[k,l,c,d] - C2_B[k,l,d,c]) *det_AA_AA
                                        pp += (1/8) * (C2_R[i,j,a,b] - C2_R[i,j,b,a]) * (C2_B[k,l,c,d] - C2_B[k,l,d,c]) *det_AA_BB
                                        pp += (1/2) * (C2_R[i,j,a,b] - C2_R[i,j,b,a]) * C2_B[k,l,c,d] *det_AA_AB
                                        pp += (1/2) * C2_R[i,j,a,b] * (C2_B[k,l,c,d] - C2_B[k,l,d,c]) *det_AB_AA
                                        pp += C2_R[i,j,a,b] * C2_B[k,l,c,d] *det_AB_AB

                                        C2_R = C2_R_pos; C2_B = C2_B_neg; disp = 1
                                        det_AA_AA = det_overlap(orbitals, [i, a+no, j, b+no], [k, c+no, l, d+no], S[disp], o, spins='AAAA')
                                        det_AA_BB = det_overlap(orbitals, [i, a+no, j, b+no], [k, c+no, l, d+no], S[disp], o, spins='AABB')
                                        det_AB_AB = det_overlap(orbitals, [i, a+no, j, b+no], [k, c+no, l, d+no], S[disp], o, spins='ABAB')
                                        det_AA_AB = det_overlap(orbitals, [i, a+no, j, b+no], [k, c+no, l, d+no], S[disp], o, spins='AAAB')
                                        det_AB_AA = det_overlap(orbitals, [i, a+no, j, b+no], [k, c+no, l, d+no], S[disp], o, spins='ABAA')
                                        pm += (1/8) * (C2_R[i,j,a,b] - C2_R[i,j,b,a]) * (C2_B[k,l,c,d] - C2_B[k,l,d,c]) *det_AA_AA
                                        pm += (1/8) * (C2_R[i,j,a,b] - C2_R[i,j,b,a]) * (C2_B[k,l,c,d] - C2_B[k,l,d,c]) *det_AA_BB
                                        pm += (1/2) * (C2_R[i,j,a,b] - C2_R[i,j,b,a]) * C2_B[k,l,c,d] *det_AA_AB
                                        pm += (1/2) * C2_R[i,j,a,b] * (C2_B[k,l,c,d] - C2_B[k,l,d,c]) *det_AB_AA
                                        pm += C2_R[i,j,a,b] * C2_B[k,l,c,d] *det_AB_AB

                                        C2_R = C2_R_neg; C2_B = C2_B_pos; disp = 2
                                        det_AA_AA = det_overlap(orbitals, [i, a+no, j, b+no], [k, c+no, l, d+no], S[disp], o, spins='AAAA')
                                        det_AA_BB = det_overlap(orbitals, [i, a+no, j, b+no], [k, c+no, l, d+no], S[disp], o, spins='AABB')
                                        det_AB_AB = det_overlap(orbitals, [i, a+no, j, b+no], [k, c+no, l, d+no], S[disp], o, spins='ABAB')
                                        det_AA_AB = det_overlap(orbitals, [i, a+no, j, b+no], [k, c+no, l, d+no], S[disp], o, spins='AAAB')
                                        det_AB_AA = det_overlap(orbitals, [i, a+no, j, b+no], [k, c+no, l, d+no], S[disp], o, spins='ABAA')
                                        mp += (1/8) * (C2_R[i,j,a,b] - C2_R[i,j,b,a]) * (C2_B[k,l,c,d] - C2_B[k,l,d,c]) *det_AA_AA
                                        mp += (1/8) * (C2_R[i,j,a,b] - C2_R[i,j,b,a]) * (C2_B[k,l,c,d] - C2_B[k,l,d,c]) *det_AA_BB
                                        mp += (1/2) * (C2_R[i,j,a,b] - C2_R[i,j,b,a]) * C2_B[k,l,c,d] *det_AA_AB
                                        mp += (1/2) * C2_R[i,j,a,b] * (C2_B[k,l,c,d] - C2_B[k,l,d,c]) *det_AB_AA
                                        mp += C2_R[i,j,a,b] * C2_B[k,l,c,d] *det_AB_AB

                                        C2_R = C2_R_neg; C2_B = C2_B_neg; disp = 3
                                        det_AA_AA = det_overlap(orbitals, [i, a+no, j, b+no], [k, c+no, l, d+no], S[disp], o, spins='AAAA')
                                        det_AA_BB = det_overlap(orbitals, [i, a+no, j, b+no], [k, c+no, l, d+no], S[disp], o, spins='AABB')
                                        det_AB_AB = det_overlap(orbitals, [i, a+no, j, b+no], [k, c+no, l, d+no], S[disp], o, spins='ABAB')
                                        det_AA_AB = det_overlap(orbitals, [i, a+no, j, b+no], [k, c+no, l, d+no], S[disp], o, spins='AAAB')
                                        det_AB_AA = det_overlap(orbitals, [i, a+no, j, b+no], [k, c+no, l, d+no], S[disp], o, spins='ABAA')
                                        mm += (1/8) * (C2_R[i,j,a,b] - C2_R[i,j,b,a]) * (C2_B[k,l,c,d] - C2_B[k,l,d,c]) *det_AA_AA
                                        mm += (1/8) * (C2_R[i,j,a,b] - C2_R[i,j,b,a]) * (C2_B[k,l,c,d] - C2_B[k,l,d,c]) *det_AA_BB
                                        mm += (1/2) * (C2_R[i,j,a,b] - C2_R[i,j,b,a]) * C2_B[k,l,c,d] *det_AA_AB
                                        mm += (1/2) * C2_R[i,j,a,b] * (C2_B[k,l,c,d] - C2_B[k,l,d,c]) *det_AB_AA
                                        mm += C2_R[i,j,a,b] * C2_B[k,l,c,d] *det_AB_AB

    elif orbitals == 'SPIN':
        for i in range(0, no, 1):
            for a in range(0, nv, 1):
                for j in range(0, no, 1):
                    for b in range(0, nv, 1):
                        for k in range(0, no, 1):
                            for c in range(0, nv, 1):
                                for l in range(0, no, 1):
                                    for d in range(0, nv, 1):
                                        C2_R = C2_R_pos; C2_B = C2_B_pos; disp = 0
                                        det = det_overlap(orbitals, [i, a+no, j, b+no], [k, c+no, l, d+no], S[disp], o)
                                        pp += (1/16) * C2_R[i,j,a,b] * C2_B[k,l,c,d] * det

                                        C2_R = C2_R_pos; C2_B = C2_B_neg; disp = 1
                                        det = det_overlap(orbitals, [i, a+no, j, b+no], [k, c+no, l, d+no], S[disp], o)
                                        pm += (1/16) * C2_R[i,j,a,b] * C2_B[k,l,c,d] * det

                                        C2_R = C2_R_neg; C2_B = C2_B_pos; disp = 2
                                        det = det_overlap(orbitals, [i, a+no, j, b+no], [k, c+no, l, d+no], S[disp], o)
                                        mp += (1/16) * C2_R[i,j,a,b] * C2_B[k,l,c,d] * det

                                        C2_R = C2_R_neg; C2_B = C2_B_neg; disp = 3
                                        det = det_overlap(orbitals, [i, a+no, j, b+no], [k, c+no, l, d+no], S[disp], o)
                                        mm += (1/16) * C2_R[i,j,a,b] * C2_B[k,l,c,d] * det

    return (((pp - pm - mp + mm)/(4*R_disp*B_disp))).imag

