if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")

import psi4
import magpy
import numpy as np

class Hessian(object):

    def __init__(self, molecule, charge=0, spin=1, method='HF'):

        # Ensure geometry remains fixed in space
        molecule.fix_orientation(True)
        molecule.fix_com(True)
        molecule.update_geometry()
        self.molecule = molecule

        self.charge = charge
        self.spin = spin
        self.method = method

        self.geom = molecule.geometry().np
        self.natom = self.geom.shape[0]


    def compute(self, disp, error_order, e_conv=1e-10, r_conv=1e-10, maxiter=400, max_diis=8, start_diis=1, print_level=0):

        # 0) Compute energy for reference geometry (E0)
        # 1) Loop over atomic x,y,z coordinates
        # 2) if error_order == 2:
        #        Compute energy for +/- displacements of the given coord
        #        Store in array of dimensions: E(natom, 3, 2)
        #    elif error_order == 4:
        #        Compute energy for +/- and 2+/2- displacements of the given coord
        #        Store in array of dimensions: E(natom, 3, 4)

        for M in range()

        findif(self, energy, 4)


    def energy(self, disp):
        """
        Wrapper energy function to feed to finite-difference functions
        """

        
