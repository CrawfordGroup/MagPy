if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")

import psi4
import magpy
import numpy as np
from .utils import shift_geom

class APT(object):

    def __init__(self, molecule, charge=0, spin=1, method='HF'):

        valid_methods = ['HF', 'CID']
        method = method.upper()
        if method not in valid_methods:
            raise Exception(f"{method:s} is not an allowed choice of method.")
        self.method = method

        # Ensure geometry remains fixed in space
        molecule.fix_orientation(True)
        molecule.fix_com(True)
        molecule.update_geometry()
        self.molecule = molecule

        self.charge = charge
        self.spin = spin

        self.geom = molecule.geometry().np
        self.natom = self.geom.shape[0]


    def compute(self, R_disp=0.001, F_disp=0.0001, e_conv=1e-10, r_conv=1e-10, maxiter=400, max_diis=8, start_diis=1, print_level=0):

        params = [e_conv, r_conv, maxiter, max_diis, start_diis, print_level]

        if print_level > 0:
            print("Initial geometry:")
            print(self.molecule.geometry().np)

#        E0 = self.energy(0, 0, 0, 0, 0, 0, params)

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
            scf = magpy.hfwfn(H, self.charge, self.spin, print_level)
            escf, C = scf.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)

#            if print_level > 0:
#                print(f"{M:d}, {alpha:d}, {beta:d} ::: {R_disp:0.5f}; {F_disp:0.5f}")
#                print(H.molecule.geometry().np)
#                print(f"ESCF = {escf:18.15f}")

            if self.method == 'HF':
                E_pos = escf
            elif self.method == 'CID':
                ci = magpy.ciwfn(scf)
                eci, C0, C2 = ci.solve_cid(e_conv, r_conv, maxiter, max_diis, start_diis)
                E_pos = eci + escf

            H = magpy.Hamiltonian(shift_geom(self.molecule, M*3+alpha, R_disp))
            H.add_field(field='electric-dipole', strength=-1.0*strength[beta])
            scf = magpy.hfwfn(H, self.charge, self.spin, print_level)
            escf, C = scf.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)

            if self.method == 'HF':
                E_neg = escf
            elif self.method == 'CID':
                ci = magpy.ciwfn(scf)
                eci, C0, C2 = ci.solve_cid(e_conv, r_conv, maxiter, max_diis, start_diis)
                E_neg = eci + escf

            mu[beta] = -(E_pos - E_neg)/(2 * F_disp)
        
        if print_level > 0:
            print("Dipole moment = ", mu)

        return mu
