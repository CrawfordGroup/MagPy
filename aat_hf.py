import psi4
from hamiltonian import Hamiltonian
from hfwfn import hfwfn
import numpy as np
from utils import *

class AAT_HF(object):

    def __init__(self, molecule, charge=0, spin=1):

        # Ensure geometry remains fixed in space
        molecule.fix_orientation(True)
        molecule.fix_com(True)
        molecule.update_geometry()
        self.molecule = molecule

        self.charge = charge
        self.spin = spin


    def shift_geom(self, R, R_disp):

        # Clone input molecule for this perturbation
        this_mol = self.molecule.clone()

        # Grab the original geometry and shift the current coordinate
        geom = np.copy(this_mol.geometry().np)
        geom[R//3][R%3] += R_disp
        geom = psi4.core.Matrix.from_array(geom) # Convert to Psi4 Matrix
        this_mol.set_geometry(geom)
        this_mol.fix_orientation(True)
        this_mol.fix_com(True)
        this_mol.update_geometry()

        return this_mol


    def compute(self, R_disp, B_disp, e_conv=1e-12, r_conv=1e-12, maxiter=100, max_diis=8, start_diis=1):

        # Unperturbed Hamiltonian
        H = Hamiltonian(self.molecule)

        # Compute the unperturbed HF wfn
        scf0 = hfwfn(H, self.charge, self.spin)
        escf0, C0 = scf0.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)

        # Occupied MO slice
        o = slice(0,scf0.ndocc)

        AAT = np.zeros((3*self.molecule.natom(), 3))

        # Loop over magnetic field displacements
        for B in range(3):
            strength = np.zeros(3)


            # +B displacement
            H.reset_V()
            strength[B] = B_disp
            H.add_field(field='magnetic-dipole', strength=strength)
            escf_B_pos, C_B_pos = scf0.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
            C_B_pos = match_phase(C0, H.basisset, C_B_pos, H.basisset)

            # -B displacement
            H.reset_V()
            strength[B] = -B_disp
            H.add_field(field='magnetic-dipole', strength=strength)
            escf_B_neg, C_B_neg = scf0.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
            C_B_neg = match_phase(C0, H.basisset, C_B_neg, H.basisset)

            # Loop over atomic coordinate displacements
            for R in range(3*self.molecule.natom()):

                # +R displacement
                H_pos = Hamiltonian(self.shift_geom(R, R_disp))
                scf_R_pos = hfwfn(H_pos, self.charge, self.spin)
                escf_R_pos, C_R_pos = scf_R_pos.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
                C_R_pos = match_phase(C0, H.basisset, C_R_pos, H_pos.basisset)

                # -R displacement
                H_neg = Hamiltonian(self.shift_geom(R, -R_disp))
                scf_R_neg = hfwfn(H_neg, self.charge, self.spin)
                escf_R_neg, C_R_neg = scf_R_neg.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
                C_R_neg = match_phase(C0, H.basisset, C_R_neg, H_neg.basisset)

                # Compute determinantal overlaps for finite-difference
                pp = det_overlap2(C_R_pos[:,o], H_pos.basisset, C_B_pos[:,o], H.basisset).imag
                pm = det_overlap2(C_R_pos[:,o], H_pos.basisset, C_B_neg[:,o], H.basisset).imag
                mp = det_overlap2(C_R_neg[:,o], H_neg.basisset, C_B_pos[:,o], H.basisset).imag
                mm = det_overlap2(C_R_neg[:,o], H_neg.basisset, C_B_neg[:,o], H.basisset).imag

                # Compute AAT element
                AAT[R,B] = ((pp - pm - mp + mm)/(2*R_disp*B_disp))

        return AAT
