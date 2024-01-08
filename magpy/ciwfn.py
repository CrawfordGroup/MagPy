class ciwfn(object):

    def __init__(self, hfwfn):

        self.hfwfn = hfwfn


        
    def solve_cid(self, e_conv=1e-7, r_conv=1e-7, maxiter=100):


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

