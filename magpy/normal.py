import psi4
import magpy
import numpy as np
from .utils import AAT_nuc

def normal(molecule):

    # Compute the Hessian [Eh/(a0^2)]
    hessian = magpy.Hessian(molecule, 0, 1, 'HF')
    disp = 0.001
    e_conv = 1e-14
    r_conv = 1e-14
    maxiter = 400
    H = hessian.compute(disp, e_conv, r_conv, maxiter)

    # Mass-weight the Hessian (Eh/(a0^2 m_e))
    _au2amu = psi4.qcel.constants.get("electron mass in u")
    masses = np.array([molecule.mass(i) * (1/_au2amu) for i in range(molecule.natom())])
    M = np.diag(1/np.sqrt(np.repeat(masses, 3)))
    H = M.T @ H @ M

    # Compute the mass-weighted normal modes
    w, Lxm = np.linalg.eigh(H)

    # (Eh/(a0^2 m_e)) (1/(2 pi c * 100)
    _c = psi4.qcel.constants.get("speed of light in vacuum")
    _hartree2J = psi4.qcel.constants.get("Hartree energy")
    _bohr2m = psi4.qcel.constants.get("Bohr radius")
    _amu2kg = psi4.qcel.constants.get("atomic mass unit-kilogram relationship")
    au2wavenumber = np.sqrt(_hartree2J/(_bohr2m * _bohr2m * _amu2kg * _au2amu)) * (1.0/(2.0 * np.pi * _c * 100.0))

    for i in range(3*molecule.natom(), 5, -1)):
        print(np.sqrt(w[i]) * au2wavenumber)

    # Remove mass-weighting from eigenvectors and extract normal vibrational modes
    Lx = M @ Lxm
"""
    APT = magpt.APT(molecule, 0, 1, 'HF')
    r_disp = 0.001
    f_disp = 0.0001
    e_conv = 1e-14
    r_conv = 1e-14
    maxiter = 400
    P = APT.compute(r_disp, f_disp, e_conv, r_conv, maxiter)

    AAT = magpy.AAT_HF(molecule)
    r_disp = 0.0001
    b_disp = 0.0001
    I = AAT.compute(r_disp, b_disp) # electronic contribution
    J = magpy.AAT_nuc(molecule) # nuclear contribution
    M = I + J
"""

    
