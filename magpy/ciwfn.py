class ciwfn(object):

    def __init__(self, hfwfn):

        self.hfwfn = hfwfn
        self.nt = hfwfn.nao
        self.no = nfwfn.ndocc
        self.nv = self.nt - self.no

        # Set up orbital subspace slices
        o = slice(0, no)
        v = slice(no, nt)
        a = slice(0, nt)

        ## Transform Hamiltonian to MO basis
        nmo = nao = self.nt # For readability
        # AO->MO two-electron integral transformation
        ERI = self.hfwfn.H.ERI
        for p in range(nao):
            for q in range(nao):
                X = ERI[p, q, a, a]
                X = C.conj().T @ X @ C
                ERI[p, q, a, a] = X

        for r in range(nmo):
            for s in range(nmo):
                X = ERI[a, a, r, s]
                X = C.conj().T @ X @ C
                ERI[:nmo, :nmo, r, s] = X

        # Convert to Dirac ordering and build spin-adapted L
        self.ERI = ERI.swapaxes(1,2)
        self.L = 2.0 * ERI - ERI.swapaxes(2,3)

        # Build MO-basis Fock matrix (diagonal for canonical MOs, but we don't assume them)
        self.h = C.conj().T @ (self.hfwfn.H.T + self.hfwfn.H.V) @ C
        self.h0 = self.h.copy() # Keep original core Hamiltonian
        self.F = self.h + contract('pmqm->pq', L[a,o,a,o])

        # Build orbital energy denominators
        eps_occ = np.diag(F)[o]
        eps_vir = np.diag(F)[v]
        Dia = eps_occ.reshape(-1,1) - eps_vir
        Dijab = eps_occ.reshape(-1,1,1,1) + eps_occ.reshape(-1,1,1) - eps_vir.reshape(-1,1) - eps_vir

        
    def solve_cid(self, e_conv=1e-7, r_conv=1e-7, maxiter=100):
        # initial guess amplitudes
        t2 = ERI[o,o,v,v]/Dijab

        # initial CISD energy (= MP2 energy)
        eci = compute_cid_energy(o, v, L, t2)

        print("CID Iter %3d: CID Ecorr = %.15f  dE = % .5E  MP2" % (0, eci, -eci))

        ediff = eci
        while ((abs(ediff) > e_conv) or (abs(rmsd) > r_conv)) and (niter <= maxiter):
            niter += 1
            eci_last = eci

            r2 = r_T2(o, v, eci, F, ERI, t2)

            t2 += r2/Dijab

            rms = contract('ijab,ijab->', r2/Dijab, r2/Dijab)
            rms = np.sqrt(rms)

            eci = compute_cid_energy(o, v, L, t2)
            ediff = eci - eci_last

            print('CID Iter %3d: CID Ecorr = %.15f  dE = % .5E  rms = % .5E' % (niter, eci, ediff, rms))


    def r_T2(o, v, E, F, ERI, t2):
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


    def compute_cid_energy(o, v, L, t2):
        eci = contract('ijab,ijab->', t2, L[o,o,v,v])
        return eci

