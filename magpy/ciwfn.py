import numpy as np
import psi4
from opt_einsum import contract
import psi4
from .utils import diis


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

    def solve_cid(self, e_conv=1e-7, r_conv=1e-7, maxiter=100, max_diis=8, start_diis=1, alg='PROJECTED'):

        valid_algs = ['PROJECTED', 'DAVIDSON']
        alg = alg.upper()
        if alg not in valid_algs:
            raise Exception("%s is not a valid choice of CI algorithm." % (alg))

        o = self.o
        v = self.v
        no = self.no
        nv = self.nv
        F = self.F
        ERI = self.ERI
        L = self.L
        Dijab = self.Dijab

        if alg == 'PROJECTED':
            # initial guess amplitudes
            C2 = ERI[o,o,v,v]/Dijab

            # initial CI energy (= MP2 energy)
            eci = self.compute_cid_energy(o, v, L, C2)

            # Setup DIIS object
            diis = diis('CI', C2, max_diis)

            print("CID Iter %3d: CID Ecorr = %.15f  dE = % .5E  MP2" % (0, eci, -eci))

            ediff = eci
            niter = 0
            rmsd = 0.0
            while ((abs(ediff) > e_conv) or (abs(rmsd) > r_conv)) and (niter <= maxiter):
                niter += 1
                eci_last = eci

                r2 = self.r_T2(o, v, eci, F, ERI, C2)
                C2 += r2/Dijab

                rms = contract('ijab,ijab->', r2/Dijab, r2/Dijab)
                rms = np.sqrt(rms)

                eci = self.compute_cid_energy(o, v, L, C2)

                ediff = eci - eci_last
                print('CID Iter %3d: CID Ecorr = %.15f  dE = % .5E  rms = % .5E' % (niter, eci, ediff, rms))

                diis.add_error_vector([C2])
                if niter >= start_diis:
                    C2 = diis.extrapolate()

        elif alg == 'DAVIDSON':
            N = M = 1 # ground state only
            maxM = 10
            sigma_done = 0
            sigma_len = no*no*nv*nv

            E = np.zeros((N))

            S = np.empty((0,sigma_len+1), float)
            C2 = ERI[o,o,v,v]/Dijab
            C = np.reshape(C2, (M, sigma_len))
            D = Dijab.flatten()

            converged = False
            for niter in range(1,maxiter+1):
                E_old = E

                Q, _ = np.linalg.qr(C.T)
                phase = np.diag((C @ Q)[:M])
                phase = np.append(phase, np.ones(Q.shape[1]-M))
                Q = phase * Q
                C = Q.T.copy()
                M = C.shape[0]

                print("CID Iter %3d: M = %3d" % (niter, M))

                # Extract guess vectors for sigma calculation
                nvecs = M - sigma_done
                C2 = np.reshape(C[sigma_done:M,:], (nvecs,no,no,nv,nv))

                # Compute sigma vectors
                s2 = np.zeros_like(C2)
                for state in range(nvecs):
                    s2[state] = self.s2(o, v, F, ERI, L, C2[state])
                sigma_done = M

                # Build and diagonalize subspace Hamiltonian
                S = np.vstack((S, np.reshape(s2, (nvecs, sigma_len))))
                G = C @ S.T
                E, a = np.linalg.eigh(G)

                # Sort eigenvalues and corresponding eigenvectors into ascending order
                idx = E.argsort()[:N]
                E = E[idx]; a = a[:,idx]

                # Build correction vectors
                r = a.T @ S - np.diag(E) @ a.T @ C
                r_norm = np.linalg.norm(r, axis=1)
                delta = r/np.subtract.outer(E,Dijab) # element-by-element division

                dE = E - E_old
                for state in range(N):
                    print("%20.12f %20.12f %20.12f" % (E[state], dE[state], r_norm[state]))

                dE = E - E_old
                print("             E/state                   dE                 norm")
                for state in range(N):
                    print("%20.12f %20.12f %20.12f" % (E[state], dE[state], r_norm[state]))

                if (np.abs(np.linalg.norm(dE)) <= e_conv):
                    converged = True
                    break

                if M >= maxM:
                    # Collapse to N vectors if subspace is too large
                    print("\nMaximum allowed subspace dimension (%d) reached. Collapsing to N roots." % (maxM))
                    C = a.T @ C
                    M = N
                    E = E_old
                    sigma_done = 0
                    S = np.empty((0,sigma_len), float)
                else:
                    # Add new vectors to guess space
                    C = np.concatenate((C, delta[:N]))

            if converged:
                #print("\nCID converged in %.3f seconds." % (time.time() - time_init))
                print("\nState     E_h           eV")
                print("-----  ------------  ------------")
                eVconv = psi4.qcel.constants.get("hartree energy in ev")
                for state in range(N):
                    print("  %3d  %12.10f  %12.10f" %(state, E[state], E[state]*eVconv))

                return E, C



    def r_T2(self, o, v, E, F, ERI, C2):
        r2 = 0.5 * ERI[o,o,v,v].copy()
        r2 += contract('ijae,be->ijab', C2, F[v,v])
        r2 -= contract('imab,mj->ijab', C2, F[o,o])
        r2 += 0.5 * contract('mnab,mnij->ijab', C2, ERI[o,o,o,o])
        r2 += 0.5 * contract('ijef,abef->ijab', C2, ERI[v,v,v,v])

        r2 += contract('imae,mbej->ijab', (C2 - C2.swapaxes(2,3)), ERI[o,v,v,o])
        r2 += contract('imae,mbej->ijab', C2, (ERI[o,v,v,o] - ERI[o,v,o,v].swapaxes(2,3)))
        r2 -= contract('mjae,mbie->ijab', C2, ERI[o,v,o,v])
        r2 += r2.swapaxes(0,1).swapaxes(2,3)
        r2 -= E*C2
        return r2


    def compute_cid_energy(self, o, v, L, t2):
        eci = contract('ijab,ijab->', t2, L[o,o,v,v])
        return eci


    def s2(self, o, v, F, ERI, L, C2):
        s2 = L[o,o,v,v].copy()
        s2 += contract('ae,ijeb->ijab', F[v,v], C2)
        s2 -= contract('mi,mjab->ijab', F[o,o], C2)
        s2 += 0.5 * contract('mnij,mnab->ijab', ERI[o,o,o,o], C2)
        s2 += 0.5 * contract('ijef,abef->ijab', C2, ERI[v,v,v,v])
        s2 += contract('imae,mbej->ijab', (C2 - C2.swapaxes(2,3)), ERI[o,v,v,o])
        s2 += contract('imae,mbej->ijab', C2, (ERI[o,v,v,o] - ERI[o,v,o,v].swapaxes(2,3)))
        s2 -= contract('mjae,mbie->ijab', C2, ERI[o,v,o,v])
#s2 -= contract('imeb,maje->ijab', C2, ERI[o,v,o,v])
#s2 -= contract('imea,mbej->ijab', C2, ERI[o,v,v,o])
#s2 += contract('miea,mbej->ijab', C2, L[o,v,v,o])

        s2 += s2.swapaxes(0,1).swapaxes(2,3)

        return s2
