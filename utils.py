import numpy as np
from opt_einsum import contract
import math

class helper_diis(object):
    def __init__(self, F, max_diis):
        self.diis_F = [F.copy()] # List of Fock matrices
        self.diis_errors = [] # List of error matrices
        self.diis_size = 0 # Current DIIS dimension
        self.max_diis = max_diis # Maximum DIIS dimension

    def add_error_vector(self, F, D, S, X):
        self.diis_F.append(F.copy())
        e = X @ (F @ D @ S - (F @ D @ S).conj().T) @ X
        self.diis_errors.append(e)

    def extrapolate(self, F):
        if(self.max_diis == 0):
            return F

        if (len(self.diis_errors) > self.max_diis):
            del self.diis_F[0]
            del self.diis_errors[0]

        self.diis_size = len(self.diis_errors)

        # Build DIIS matrix B
        B = -1 * np.ones((self.diis_size + 1, self.diis_size + 1), dtype=type(F[0,0]))
        B[-1, -1] = 0

        for n1, e1 in enumerate(self.diis_errors):
            B[n1, n1] = contract('pq,pq->', e1.conjugate(), e1)
            for n2, e2 in enumerate(self.diis_errors):
                if n1 > n2:
                    continue
                B[n1, n2] = contract('pq,pq->', e1.conjugate(), e2)
                B[n2, n1] = B[n1, n2]


#        B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()
        A = np.zeros((self.diis_size+1), dtype=type(F[0,0]))
        A[-1] = -1

        c = np.linalg.solve(B, A)

        F[:,:] = 0
        for i in range(self.diis_size):
            F = F + c[i] * self.diis_F[i+1]

        return F

#def distance(v, u):
#    """Compute the distance between points defined by vectors *v* and *u*."""
#    return math.sqrt(sum(((v[i] - u[i]) * (v[i] - u[i]) for i in range(v.size))))
#
