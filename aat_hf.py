import psi4

class AAT_HF(object):

    def __init__(self, molecule, charge=0, spin=1):

        # Ensure geometry remains fixed in space
        molecule.fix_orientation(True)
        molecule.fix_com(True)
        molecule.update_geometry()
        self.molecule = molecule

        self.charge = charge
        self.spin = spin

    def compute(self, R_disp, B_disp, e_conv=1e-12, r_conv=1e-12, maxiter=100, max_diis=8, start_diis=1):

        # Prepare unperturbed Hamiltonian -- used for unperturbed and mag-field perturbations
        H = Hamiltonian(self.molecule)

        # Compute the unperturbed HF wfn for reference -- also used for mag-field perturbations
        scf0 = hfwfn(H, self.charge, self.spin)
        escf0, C0 = scf0.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
        basis0 = psi4.core.BasisSet.build(self.molecule)

        # Hang on to the original geometry as NumPy array
        self.geom0 = self.molecule.geometry().np

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
            for R in range(3*self.molecule.natom()):

                # Clone of input molecule for this perturbation
                molecule = self.molecule.clone()

                # Grab the original geometry and shift the current coordinate
                geom = np.copy(this_molecule.geometry().np)
                geom[R//3][R%3] += R_disp
                geom = psi4.core.Matrix.from_array(geom) # Convert to Psi4 Matrix
                this_molecule.set_geometry(geom)
                this_molecule.fix_orientation(True)
                this_molecule.fix_com(True)
                this_molecule.update_geometry()
                this_basis = psi4.core.BasisSet.build(this_molecule)

                this_H = Hamiltonian(self.molecule)
                
                scf_R_pos = hfwfn(this_H, self.charge, self.spin)
                escf_R_pos, C_R_pos = scf.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
                match_phase(C0, basis0, C_R_pos, this_basis)
