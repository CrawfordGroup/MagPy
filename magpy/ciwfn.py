if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")

import numpy as np
import psi4
from opt_einsum import contract
import psi4
from .utils import DIIS


class ciwfn(object):

    def __init__(self, hfwfn):

        self.hfwfn = hfwfn

        self.print_level = hfwfn.print_level

        nfzc = hfwfn.H.basisset.n_frozen_core()
        nt = self.nt = hfwfn.nao - nfzc # assumes number of MOs = number of AOs
        no = self.no = hfwfn.ndocc - nfzc
        nv = self.nv = hfwfn.nao - self.no - nfzc

        if self.print_level > 0:
            print("\nNMO = %d; NACT = %d; NO = %d; NV = %d" % (hfwfn.nao, self.nt, self.no, self.nv))

        # Set up orbital subspace slices
        o = self.o = slice(0, no)
        v = self.v = slice(no, nt)
        a = self.a = slice(0, nt)

        ## Transform Hamiltonian to MO basis

        # AO-basis one-electron Hamiltonian
        h = self.hfwfn.H.T + self.hfwfn.H.V

        # If there are frozen core orbitals, build and add the frozen-core operator 
        # (core contribution to Fock operator) to the one-electron Hamiltonian
        efzc = 0
        if nfzc > 0:
            C = self.hfwfn.C[:,:nfzc] # only core MOs
            Pc = contract('pi,qi->pq', C.conj(), C)
            ERI = self.hfwfn.H.ERI
            hc = h + 2.0 * contract('pqrs,pq->rs', ERI, Pc) - contract('pqrs,ps->qr', ERI, Pc)
            efzc = contract('pq,pq->', (h+hc), Pc)
            h = hc

        # Select active MOs
        C = self.hfwfn.C[:,nfzc:]

        # AO->MO one-electron integral transformation
        self.h = C.conj().T @ h @ C
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
        ESCF = efzc + 2.0 * contract('ii->',self.h[o,o]) + contract('ijij->', L[o,o,o,o])
        if self.print_level > 0:
            print("ESCF (electronic) = ", ESCF)
            print("ESCF (total) =      ", ESCF+self.hfwfn.H.enuc)
        self.E0 = ESCF+self.hfwfn.H.enuc

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
        E0 = self.E0
        F = self.F
        ERI = self.ERI
        L = self.L
        Dijab = self.Dijab

        if alg == 'PROJECTED':
            if self.print_level > 0:
                print("\nSolving projected CID equations.")

            # initial guess amplitudes
            C2 = ERI[o,o,v,v]/Dijab

            # initial CI energy (= MP2 energy)
            eci = self.compute_cid_energy(o, v, L, C2)

            # Setup DIIS object
            diis = DIIS(C2, max_diis)

            if self.print_level > 0:
                print("CID Iter %3d: CID Ecorr = %.15f  dE = % .5E  MP2" % (0, eci, -eci))

            ediff = eci
            rmsd = 0.0
            for niter in range(1, maxiter+1):
                eci_last = eci

                r2 = self.r_T2(o, v, eci, F, ERI, L, C2)
                C2 += r2/Dijab

                rms = contract('ijab,ijab->', r2/Dijab, r2/Dijab)
                rms = np.sqrt(rms)

                eci = self.compute_cid_energy(o, v, L, C2)

                ediff = eci - eci_last

                if self.print_level > 0:
                    print('CID Iter %3d: CID Ecorr = %.15f  dE = % .5E  rms = % .5E' % (niter, eci, ediff, rms))

                if ((abs(ediff) < e_conv) and (abs(rms) < r_conv)):
                    if self.print_level > 0:
                        print("\nCID Equations converged.")
                        print("CID Correlation Energy = %.15f" % (eci))
                        print("CID Total Energy       = %.15f" % (eci + E0))
                    return eci, C2

                diis.add_error_vector(C2, r2/Dijab)
                if niter >= start_diis:
                    C2 = diis.extrapolate(C2)

        elif alg == 'DAVIDSON':
            if self.print_level > 0:
                print("\nSolving CID equations using Davidson algorithm.")

            N = M = 1 # ground state only
            maxM = 10
            sigma_done = 0
            sigma_len = no*no*nv*nv

            E = np.zeros((N))

            S = np.empty((0,sigma_len+1), float)
            C = np.empty((0,sigma_len+1), float)

            C0 = 1.0
            C2 = ERI[o,o,v,v]/Dijab
            E[0] = self.compute_cid_energy(o, v, L, C2)

            C = np.vstack((C, np.hstack((C0, C2.flatten()))))
            D = np.hstack((E0, Dijab.flatten()))

            converged = False
            for niter in range(1,maxiter+1):
                E_old = E

                Q, _ = np.linalg.qr(C.T)
                phase = np.diag((C @ Q)[:M])
                phase = np.append(phase, np.ones(Q.shape[1]-M))
                Q = phase * Q
                C = Q.T.copy()
                M = C.shape[0]

                # Extract guess vectors for sigma calculation
                nvecs = M - sigma_done
                C0 = C[sigma_done:M, 0]
                C2 = np.reshape(C[sigma_done:M,1:sigma_len+1], (nvecs,no,no,nv,nv))

                # Compute sigma vectors
                s0 = np.zeros(nvecs)
                s2 = np.zeros_like(C2)
                for state in range(nvecs):
                    s0[state], s2[state] = self.sigma(o, v, F, ERI, L, C0[state], C2[state])
                sigma_done = M

                # Build and diagonalize subspace Hamiltonian
                S = np.vstack((S, np.hstack((np.reshape(s0, (nvecs,1)), np.reshape(s2, (nvecs, sigma_len))))))
                G = C @ S.T
                E, a = np.linalg.eigh(G)

                # Sort eigenvalues and corresponding eigenvectors into ascending order
                idx = E.argsort()[:N]
                E = E[idx]; a = a[:,idx]

                # Build correction vectors
                r = a.T @ S - np.diag(E) @ a.T @ C
                r_norm = np.linalg.norm(r, axis=1)
                delta = r/np.subtract.outer(E,D) # element-by-element division

                dE = E - E_old
                for state in range(N):
                    if self.print_level > 0:
                        print("%20.12f %20.12f %20.12f" % (E[state], dE[state], r_norm[state]))

                dE = E - E_old
                if self.print_level > 0:
                    print('CID Iter %3d: CID Ecorr = %.15f  dE = % .5E  rms = % .5E  C0 = %.10f' % (niter, E[0], dE[0], r_norm[0], C0[0]))

                if (np.abs(np.linalg.norm(dE)) <= e_conv):
                    converged = True
                    break

                if M >= maxM:
                    # Collapse to N vectors if subspace is too large
                    if self.print_level > 0:
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
                eVconv = psi4.qcel.constants.get("hartree energy in ev")
                if self.print_level > 0:
                    print("\nState     E_h           eV")
                    print("-----  ------------  ------------")
                    for state in range(N):
                        print("  %3d  %12.10f  %12.10f" %(state, E[state], E[state]*eVconv))

                return E, C



    def r_T2(self, o, v, E, F, ERI, L, C2):
        r2 = 0.5 * ERI[o,o,v,v].copy()
        r2 += contract('ijae,be->ijab', C2, F[v,v])
        r2 -= contract('imab,mj->ijab', C2, F[o,o])
        r2 += 0.5 * contract('mnab,mnij->ijab', C2, ERI[o,o,o,o])
        r2 += 0.5 * contract('ijef,abef->ijab', C2, ERI[v,v,v,v])

        r2 -= contract('imeb,maje->ijab', C2, ERI[o,v,o,v])
        r2 -= contract('imea,mbej->ijab', C2, ERI[o,v,v,o])
        r2 += contract('miea,mbej->ijab', C2, L[o,v,v,o])

        r2 += r2.swapaxes(0,1).swapaxes(2,3)
        r2 -= E*C2
        return r2


    def compute_cid_energy(self, o, v, L, t2):
        eci = contract('ijab,ijab->', t2, L[o,o,v,v])
        return eci


    def sigma(self, o, v, F, ERI, L, C0, C2):
        s0 = contract('ijab,ijab->', C2, ERI[o,o,v,v])

        s2 = 0.5 * C0 * ERI[o,o,v,v].copy()
        s2 += contract('ae,ijeb->ijab', F[v,v], C2)
        s2 -= contract('mi,mjab->ijab', F[o,o], C2)
        s2 += 0.5 * contract('mnij,mnab->ijab', ERI[o,o,o,o], C2)
        s2 += 0.5 * contract('ijef,abef->ijab', C2, ERI[v,v,v,v])
        s2 -= contract('imeb,maje->ijab', C2, ERI[o,v,o,v])
        s2 -= contract('imea,mbej->ijab', C2, ERI[o,v,v,o])
        s2 += contract('miea,mbej->ijab', C2, L[o,v,v,o])

        s2 += s2.swapaxes(0,1).swapaxes(2,3)
        s2 += E * C2

        return s0, s2

