if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")

import psi4
import magpy
import numpy as np
from .utils import shift_geom

class Hessian(object):

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


    def compute(self, disp=0.001, e_conv=1e-10, r_conv=1e-10, maxiter=400, max_diis=8, start_diis=1, print_level=0):

        params = [e_conv, r_conv, maxiter, max_diis, start_diis, print_level]

        if print_level > 0:
            print("Initial geometry:")
            print(self.molecule.geometry().np)

        E0 = self.energy(0, 0, 0, 0, 0, 0, params)

        hess = np.zeros((self.natom*3, self.natom*3))
        for R in range(self.natom*3):
            for S in range(R+1):
                M1 = R//3; alpha1 = R%3 # left-hand atom and coordinate
                M2 = S//3; alpha2 = S%3 # right-hand atom and coordinate

                if R != S:
                    Epp = self.energy(M1, alpha1, disp, M2, alpha2, disp, params)
                    Epm = self.energy(M1, alpha1, disp, M2, alpha2, -disp, params)
                    Emp = self.energy(M1, alpha1, -disp, M2, alpha2, disp, params)
                    Emm = self.energy(M1, alpha1, -disp, M2, alpha2, -disp, params)

                    hess[R,S] = hess[S,R] = (Epp - Epm - Emp + Emm)/(4*disp*disp)
                else:
                    E2p = self.energy(M1, alpha1, 2*disp, M2, alpha2, 0, params)
                    Ep = self.energy(M1, alpha1, disp, M2, alpha2, 0, params)
                    Em = self.energy(M1, alpha1, -disp, M2, alpha2, 0, params)
                    E2m = self.energy(M1, alpha1, -2*disp, M2, alpha2, 0, params)

                    hess[R,R] = -(E2p - 16*Ep + 30*E0 - 16*Em + E2m)/(12*disp*disp)

        if print_level > 0:
            print("Hessian (Eh/a0^2)")
            print(hess)
        return hess


    def energy(self, M1, alpha1, disp1, M2, alpha2, disp2, params):
        """
        Energy wrappter function
        """
        e_conv = params[0]
        r_conv = params[1]
        maxiter = params[2]
        max_diis = params[3]
        start_diis = params[4]
        print_level = params[5]

        H = magpy.Hamiltonian(shift_geom(shift_geom(self.molecule, M1*3+alpha1, disp1), M2*3+alpha2, disp2))
        scf = magpy.hfwfn(H, self.charge, self.spin, print_level)
        escf, C = scf.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
        if print_level > 0:
            print(f"{M1:d}, {alpha1:d}; {M2:d}, {alpha2:d} ::: {disp1:0.5f}; {disp2:0.5f}")
            print(H.molecule.geometry().np)
            print(f"ESCF = {escf:18.15f}")

        if self.method == 'HF':
            return escf
        elif self.method == 'CID':
            ci = magpy.ciwfn(scf)
            eci, C0, C2 = ci.solve_cid(e_conv, r_conv, maxiter, max_diis, start_diis)
            return eci + escf
