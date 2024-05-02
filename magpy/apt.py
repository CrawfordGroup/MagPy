if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")

import psi4
import magpy
import numpy as np
from .utils import shift_geom

class APT(object):

    def __init__(self, molecule, charge=0, spin=1):

        # Ensure geometry remains fixed in space
        molecule.fix_orientation(True)
        molecule.fix_com(True)
        molecule.update_geometry()
        self.molecule = molecule

        self.charge = charge
        self.spin = spin

        self.geom = molecule.geometry().np
        self.natom = self.geom.shape[0]


    def compute(self, method='HF', R_disp=0.001, F_disp=0.0001, **kwargs):

        valid_methods = ['HF', 'CID']
        method = method.upper()
        if method not in valid_methods:
            raise Exception(f"{method:s} is not an allowed choice of method.")
        self.method = method

        # Extract kwargs
        e_conv = kwargs.pop('e_conv', 1e-10)
        r_conv = kwargs.pop('r_conv', 1e-10)
        maxiter = kwargs.pop('maxiter', 400)
        max_diis = kwargs.pop('max_diis', 8)
        start_diis = kwargs.pop('start_diis', 1)
        print_level = kwargs.pop('print_level', 0)

        params = [e_conv, r_conv, maxiter, max_diis, start_diis, print_level]

        if print_level > 0:
            print("Initial geometry:")
            print(self.molecule.geometry().np)

        dipder = np.zeros((self.natom*3, 3))
        for R in range(self.natom*3):
            M = R//3; alpha = R%3 # atom and coordinate

            mu_p = self.dipole(M, alpha,  R_disp, F_disp, params)
            mu_m = self.dipole(M, alpha, -R_disp, F_disp, params)

            dipder[R] = (mu_p - mu_m)/(2*R_disp)

        if print_level > 0:
            print("APT (Eh/(e a0^2))")
            print(dipder)

        return dipder


    def dipole(self, M, alpha, R_disp, F_disp, params):
        """
        Energy wrappter function
        """
        e_conv = params[0]
        r_conv = params[1]
        maxiter = params[2]
        max_diis = params[3]
        start_diis = params[4]
        print_level = params[5]

        mu = np.zeros((3))
        strength = np.eye(3) * F_disp
        for beta in range(3):
            H = magpy.Hamiltonian(shift_geom(self.molecule, M*3+alpha, R_disp))
            H.add_field(field='electric-dipole', strength=strength[beta])
            scf = magpy.hfwfn(H, self.charge, self.spin)
            escf, C = scf.solve(e_conv=e_conv, r_conv=r_conv, maxiter=maxiter, max_diis=max_diis, start_diis=start_diis, print_level=print_level)

            if self.method == 'HF':
                E_pos = escf
            elif self.method == 'CID':
                ci = magpy.ciwfn(scf)
                eci, C0, C2 = ci.solve(e_conv=e_conv, r_conv=r_conv, maxiter=maxiter, max_diis=max_diis, start_diis=start_diis, print_level=print_level)
                E_pos = eci + escf

            H = magpy.Hamiltonian(shift_geom(self.molecule, M*3+alpha, R_disp))
            H.add_field(field='electric-dipole', strength=-1.0*strength[beta])
            scf = magpy.hfwfn(H, self.charge, self.spin)
            escf, C = scf.solve(e_conv=e_conv, r_conv=r_conv, maxiter=maxiter, max_diis=max_diis, start_diis=start_diis, print_level=print_level)

            if self.method == 'HF':
                E_neg = escf
            elif self.method == 'CID':
                ci = magpy.ciwfn(scf)
                eci, C0, C2 = ci.solve(e_conv=e_conv, r_conv=r_conv, maxiter=maxiter, max_diis=max_diis, start_diis=start_diis, print_level=print_level)
                E_neg = eci + escf

            mu[beta] = -(E_pos - E_neg)/(2 * F_disp)

        if print_level > 0:
            print("Dipole moment = ", mu)

        return mu
