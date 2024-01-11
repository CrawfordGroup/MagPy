import numpy as np
from opt_einsum import contract


class ciwfn(object):

    def __init__(self, hfwfn):

        self.hfwfn = hfwfn
        nt = self.nt = hfwfn.nao
        no = self.no = hfwfn.ndocc
        nv = self.nv = self.nt - self.no

        # Set up orbital subspace slices
        o = self.o = slice(0, no)
        v = self.v = slice(no, nt)
        a = self.a = slice(0, nt)

        ## Transform Hamiltonian to MO basis
        C = self.hfwfn.C

        # AO->MO one-electron integral transformation
        self.h = C.conj().T @ (self.hfwfn.H.T + self.hfwfn.H.V) @ C
        self.h0 = self.h.copy() # Keep original core Hamiltonian

        # AO->MO two-electron integral transformation
        ERI = self.hfwfn.H.ERI
        ERI = contract('pqrs,sl->pqrl', ERI, C)
        ERI = contract('pqrl,rk->pqkl', ERI, C.conj())
        ERI = contract('pqkl,qj->pjkl', ERI, C)
        ERI = contract('pjkl,pi->ijkl', ERI, C.conj())

        # Convert to Dirac ordering and build spin-adapted L
        ERI = self.ERI = ERI.swapaxes(1,2)
        L = self.L = 2.0 * ERI - ERI.swapaxes(2,3)

        # Build MO-basis Fock matrix (diagonal for canonical MOs, but we don't assume them)
        F = self.F = self.h + contract('pmqm->pq', L[a,o,a,o])

        # SCF check
        ESCF = 2*contract('ii->',self.h[o,o])+contract('ijij->', L[o,o,o,o])
        print("ESCF (electronic) = ", ESCF)
        print("ESCF (total) =      ", ESCF+self.hfwfn.H.enuc)

        # Build orbital energy denominators
        eps_occ = np.diag(F)[o]
        eps_vir = np.diag(F)[v]
        Dia = eps_occ.reshape(-1,1) - eps_vir # For later when I add singles
        Dijab = eps_occ.reshape(-1,1,1,1) + eps_occ.reshape(-1,1,1) - eps_vir.reshape(-1,1) - eps_vir
        self.Dijab = Dijab

    def solve_cid(self, e_conv=1e-7, r_conv=1e-7, maxiter=100, alg='PROJECT'):

        valid_algs = ['PROJECT', 'DAVIDSON']
        alg = alg.upper()
        if alg not in valid_alg:
            raise Exception("%s is not a valid choice of CI algorithm." % (alg))

        o = self.o
        v = self.v
        no = self.no
        nv = self.nv
        F = self.F
        ERI = self.ERI
        L = self.L
        Dijab = self.Dijab

        if alg == 'PROJECT':
            # initial guess amplitudes
            t2 = ERI[o,o,v,v]/Dijab

            # initial CI energy (= MP2 energy)
            eci = self.compute_cid_energy(o, v, L, t2)

            print("CID Iter %3d: CID Ecorr = %.15f  dE = % .5E  MP2" % (0, eci, -eci))

            ediff = eci
            niter = 0
            rmsd = 0.0
            while ((abs(ediff) > e_conv) or (abs(rmsd) > r_conv)) and (niter <= maxiter):
                niter += 1
                eci_last = eci

                r2 = self.r_T2(o, v, eci, F, ERI, t2)

                t2 += r2/Dijab

                rms = contract('ijab,ijab->', r2/Dijab, r2/Dijab)
                rms = np.sqrt(rms)

                eci = self.compute_cid_energy(o, v, L, t2)
                ediff = eci - eci_last

                print('CID Iter %3d: CID Ecorr = %.15f  dE = % .5E  rms = % .5E' % (niter, eci, ediff, rms))

        elif alg == 'DAVIDSON':

            N = M = 1 # ground state only
            maxM = 10
            sigma_done = 0
            sigma_len = no*no*nv*nv
            E = np.zeros((N))
            S = np.empty((0,sigma_len), float)

            # initial guess
            C = ERI[o,o,v,v]/Dijab

            # preconditioner
            Dijab = Dijab.flatten()

            converged = False
            for niter in range(1,maxiter+1):
                E_old = E

                Q, _ = np.linalg.qr(C.T)


    def r_T2(self, o, v, E, F, ERI, t2):
        r_T2 = 0.5 * ERI[o,o,v,v].copy()
        r_T2 += contract('ijae,be->ijab', t2, F[v,v])
        r_T2 -= contract('imab,mj->ijab', t2, F[o,o])
        r_T2 += 0.5 * contract('mnab,mnij->ijab', t2, ERI[o,o,o,o])
        r_T2 += 0.5 * contract('ijef,abef->ijab', t2, ERI[v,v,v,v])

        r_T2 += contract('imae,mbej->ijab', (t2 - t2.swapaxes(2,3)), ERI[o,v,v,o])
        r_T2 += contract('imae,mbej->ijab', t2, (ERI[o,v,v,o] - ERI[o,v,o,v].swapaxes(2,3)))
        r_T2 -= contract('mjae,mbie->ijab', t2, ERI[o,v,o,v])
        r_T2 += r_T2.swapaxes(0,1).swapaxes(2,3)
        r_T2 -= E*t2
        return r_T2


    def compute_cid_energy(self, o, v, L, t2):
        eci = contract('ijab,ijab->', t2, L[o,o,v,v])
        return eci


