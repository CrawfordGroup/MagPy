if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")


import psi4
import numpy as np


class Hamiltonian(object):
    """
    A molecular Hamiltonian object in the atomic orbital basis.

    Attributes
    ----------
    """
    def __init__(self, molecule):

        self.molecule = molecule
        self.basisset = psi4.core.BasisSet.build(molecule)
        mints = psi4.core.MintsHelper(self.basisset)

        self.S = np.asarray(mints.ao_overlap()) # (p|q)
        self.T = np.asarray(mints.ao_kinetic()) # (p|T|q)
        self.V = np.asarray(mints.ao_potential()) # (p|v|q)
        self.ERI = np.asarray(mints.ao_eri()) # (pr|qs)

        # Save the true nuclear-electron attraction potential in case the
        # user adds external fields later
        self.V0 = self.V 

		# Nuclear repulsion energy (zero field)
        self.enuc = self.molecule.nuclear_repulsion_energy()

        ## One-electron property integrals for adding multipole fields

        # Electric dipole integrals (length): -e r
        self.mu = mints.so_dipole()
        for i in range(3):
            self.mu[i] = np.asarray(self.mu[i])

        # Magnetic dipole integrals: -(e/2 m_e) L
        self.m = mints.ao_angular_momentum()
        for i in range(3):
            self.m[i] = -0.5j * np.asarray(self.m[i])

        # Linear momentum integrals: (-e) (-i hbar) Del
        self.p = mints.ao_nabla()
        for i in range(3):
            self.p[i] = 1.0j * np.asarray(self.p[i])

        # Traceless quadrupole: -e Q
        self.Q = mints.ao_traceless_quadrupole()
        for i in range(len(self.Q)):
            self.Q[i] = np.asarray(self.Q[i])


    def add_field(self, **kwargs):

        # Suppress printing by default
        print_level = kwargs.pop('print', 0)

        # Add predefined field or user-defined field
        # Predefined = multipole-type = - strength * mulipole integrals
        # User-defined = any function * one-electron integral field

        valid_fields = ['ELECTRIC-DIPOLE', 'ELECTRIC-DIPOLE-VELOCITY', 'ELECTRIC-QUADRUPOLE', 'MAGNETIC-DIPOLE', 'CUSTOM']
        self.field_type = kwargs.pop('field','ELECTRIC-DIPOLE').upper()
        if self.field_type not in valid_fields:
            raise Exception("%s is not a valid external field." % (self.field_type))

        if self.field_type == 'ELECTRIC-DIPOLE':
            field = self.mu
        elif self.field_type == 'MAGNETIC-DIPOLE':
            field = self.m
        elif self.field_type == 'ELECTRIC-DIPOLE-VELOCITY':
            field = self.p
        elif self.field_type == 'ELECTRIC-QUADRUPOLE':
            field = self.Q
        else:
            raise Exception("Can't handle custom fields yet.  Stay tuned!")

        self.field_strength = kwargs.pop('strength')
        if not isinstance(self.field_strength, np.ndarray):
            raise Exception("Field strength must be given as a length-3 NumPy array of type float64.")
        if self.field_strength.size != 3:
            raise Exception("Field strength must be given as a length-3 NumPy array of type float64.")
        if self.field_strength.dtype != 'float64':
            raise Exception("Field strength must be given as a length-3 NumPy array of type float64.")

        if print_level > 0:
            print(f"\n  Field type:     {self.field_type}")
            print(f"  Field strength: {self.field_strength}")

        # Add the external field to the one-electron potential
        for i in range(3):
            self.V = self.V - self.field_strength[i] * field[i]

        # Get the new nuclear repulsion energy including the field
        # NB: The libmints Molecule class *adds* the field*nuclear_dipoles to
        # the energy, so we have to multiply the field_strength by -1 for
        # consistency.
        if self.field_type == 'ELECTRIC-DIPOLE':
            self.enuc = self.molecule.nuclear_repulsion_energy([-1.0 * x for x in self.field_strength])

    def reset_V(self):

        self.V = self.V0

