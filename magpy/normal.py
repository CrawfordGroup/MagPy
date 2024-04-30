import psi4
import magpy
import numpy as np
from .utils import AAT_nuc
from opt_einsum import contract

def normal(molecule, read_hessian=False, **kwargs):

    # Physical constants and a few derived units
    _c = psi4.qcel.constants.get("speed of light in vacuum") # m/s
    _me = psi4.qcel.constants.get("electron mass") # kg
    _na = psi4.qcel.constants.get("Avogadro constant") # 1/mol
    _e = psi4.qcel.constants.get("atomic unit of charge") # C
    _e0 = psi4.qcel.constants.get("electric constant") # F/m = s^4 A^2/(kg m^3)
    _h = psi4.qcel.constants.get("Planck constant") # J s
    _hbar = _h/(2*np.pi) # J s/rad
    _mu0 = 1/(_c**2 * _e0) # s^2/(m F) = kg m/(s^2 A^2)
    _ke = 1/(4 * np.pi * _e0) # kg m^3/(C^2 s^2)
    _alpha = _ke * _e**2/(_hbar * _c) # dimensionless
    _a0 = _hbar/(_me * _c * _alpha) # m
    _Eh = (_hbar**2)/(_me * _a0**2) # J
    _u = 1/(1000 * _na) # kg
    _D = 1/(10**(21) * _c) # C m/Debye
    _bohr2angstroms = _a0 * 10**(10)

    # Frequencies in au --> cm^-1
    conv_freq_au2wavenumber = np.sqrt(_Eh/(_a0 * _a0 * _me)) * (1.0/(2.0 * np.pi * _c * 100.0))
    # IR intensities in au --> km/mol
    conv_ir_au2kmmol = (_e**2 * _ke * _na * np.pi)/(1000 * 3 * _me * _c**2)
    # VCD rotatory strengths in au --> (esu cm)**2 * (10**(44))
    conv_vcd_au2cgs = (_e**2 * _hbar * _a0)/(_me * _c) * (1000 * _c)**2 * (10**(44))

    # Compute the Hessian [Eh/(a0^2)]
    if read_hessian is False:
        hessian = magpy.Hessian(molecule, 0, 1, 'HF')
        disp = 0.001
        e_conv = 1e-13
        r_conv = 1e-13
        maxiter = 400
        max_diis = 8
        start_diis = 1
        print_level = 1
        H = hessian.compute(disp, e_conv, r_conv, maxiter, max_diis, start_diis, print_level)
    else:
        H = np.genfromtxt(kwargs.pop('file'), skip_header=1).reshape(3*molecule.natom(),3*molecule.natom())

    # Mass-weight the Hessian (Eh/(a0^2 m_e))
    masses = np.array([(molecule.mass(i)*_u/_me) for i in range(molecule.natom())])
    M = np.diag(1/np.sqrt(np.repeat(masses, 3)))
    H = M.T @ H @ M

    # Compute the mass-weighted normal modes
    w, Lxm = np.linalg.eigh(H)

    # Mass-weight eigenvectors and extract normal vibrational modes
    Lx = M @ Lxm # 1/sqrt(m_e)
    S = np.flip(Lx, 1)[:,:(3*molecule.natom()-6)] 
    freq = np.flip(np.sqrt(w))[:(3*molecule.natom()-6)]

    for i in range(3*molecule.natom()-6): # Assuming non-linear molecules for now
        print(f"{freq[i]*conv_freq_au2wavenumber:7.2f}")

    # Compute APTs and transform to normal mode basis
    APT = magpy.APT(molecule, 0, 1, 'HF')
    r_disp = 0.001
    f_disp = 0.0001
    e_conv = 1e-13
    r_conv = 1e-13
    maxiter = 400
    P = APT.compute(r_disp, f_disp, e_conv, r_conv, maxiter) # 3N x 3
    # (e a0)/(a0 sqrt(m_e))
    P = P.T @ S # 3 x (3N-6)

    # Compute IR intensities (e^2/m_e)
    ir_intensities = np.zeros((3*molecule.natom()-6))
    for i in range(3*molecule.natom()-6):
        ir_intensities[i] = contract('j,j->', P[:,i], P[:,i])

    for i in range(3*molecule.natom()-6): # Assuming non-linear molecules for now
        print(f"{freq[i]*conv_freq_au2wavenumber:7.2f} {ir_intensities[i]*conv_ir_au2kmmol:7.3f}")

    # Compute AATs and transform to normal mode basis
    AAT = magpy.AAT_HF(molecule)
    r_disp = 0.0001
    b_disp = 0.0001
    I = AAT.compute(r_disp, b_disp) # electronic contribution
    J = AAT_nuc(molecule) # nuclear contribution
    M = I + J   # 3N x 3
    M = M.T @ S # 3 x (3N-6)

    # Compute VCD rotatory strengths
    rotatory_strengths = np.zeros((3*molecule.natom()-6))
    for i in range(3*molecule.natom()-6):
        rotatory_strengths[i] = contract('j,j->', P[:,i], M[:,i])

    for i in range(3*molecule.natom()-6): # Assuming non-linear molecules for now
        print(f"{freq[i]*conv_freq_au2wavenumber:7.2f} {ir_intensities[i]*conv_ir_au2kmmol:7.3f}  {rotatory_strengths[i] * conv_vcd_au2cgs:7.3f}")

