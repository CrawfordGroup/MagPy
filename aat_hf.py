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

        # Unperturbed Hamiltonian
        H = Hamiltonian(self.molecule)

        # Compute the unperturbed HF wfn
        scf0 = hfwfn(H, self.charge, self.spin)
        escf0, C0 = scf0.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)

        # Loop over magnetic field displacements
        for B in range(3):
            strength = [0,0,0]

            # +B displacement
            H.reset_V()
            H.add_field(field='magnetic-dipole', strength=B_disp)
            escf_B_pos, C_B_pos = scf.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
            match_phase(C0, H.basisset, C_B_pos, H.basisset)

            # -B displacement
            H.reset_V()
            H.add_field(field='magnetic-dipole', strength=-B_disp)
            escf_B_neg, C_B_neg = scf.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
            match_phase(C0, H.basisset, C_B_neg, H.basisset)

            # Loop over atomic coordinate displacements
            for R in range(3*self.molecule.natom()):

                # Clone of input molecule for this perturbation
                mol_pos = self.molecule.clone()

                # Grab the original geometry and shift the current coordinate
                geom = np.copy(mol_pos.geometry().np)
                geom[R//3][R%3] += R_disp
                geom = psi4.core.Matrix.from_array(geom) # Convert to Psi4 Matrix
                mol_pos.set_geometry(geom)
                mol_pos.fix_orientation(True)
                mol_pos.fix_com(True)
                mol_pos.update_geometry()

                H_pos = Hamiltonian(mol_pos)
                
                scf_R_pos = hfwfn(H_pos, self.charge, self.spin)
                escf_R_pos, C_R_pos = scf.solve_scf(e_conv, r_conv, maxiter, max_diis, start_diis)
                match_phase(C0, basis0, C_R_pos, H_pos.basisset)
