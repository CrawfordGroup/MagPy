import psi4

class AAT_HF(object):

    def __init__(self, molecule, charge=0, spin=1):

        # Ensure geometry remains fixed in space
        molecule.fix_orientation(True)
        molecule.fix_com(True)
        molecule.update_geometry()
        self.molecule = molecule

        # Prepare unperturbed Hamiltonian
        self.H = Hamiltonian(molecule)

    def compute(self, R_disp, B_disp, e_conv=1e-12, r_conv=1e-12, maxiter=100, max_diis=8, start_diis=1):

        H = self.H

        # Compute the unperturbed HF wfn for reference
        scf0 = hfwfn(H)
        escf0, C0 = scf0.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
        basis0 = psi4.core.BasisSet.build(self.molecule)

        # Loop over magnetic field displacements
        for B in range(3):

            # +B displacement
            H.reset_V()
            H.add_field(field='magnetic-dipole', strength=B_disp)
            escf_B_pos, C_B_pos = scf.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
            match_phase(C0, basis0, C_B_pos, basis0)

            # -B displacement
            H.reset_V()
            H.add_field(field='magnetic-dipole', strength=-B_disp)
            escf_B_neg, C_B_neg = scf.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
            match_phase(C0, basis0, C_B_neg, basis0)

            # Loop over atomic coordinate displacements
            geom = self.moleculae.geometry().np
            for R in range(3*self.molecule.natom()):
