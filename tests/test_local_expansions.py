




import numpy as np
import ctypes
from scipy.special import lpmv
import math
import cmath

from coulomb_mc import mc_expansion_tools

REAL = ctypes.c_double



def re_lm(l,m): return (l**2) + l + m
def im_lm(l,m): return (l**2) + l +  m + llimit**2

def compute_phi_local(llimit, moments, disp_sph):

    phi_sph_re = 0.
    phi_sph_im = 0.
    def re_lm(l,m): return (l**2) + l + m
    def im_lm(l,m): return (l**2) + l +  m + llimit**2

    for lx in range(llimit):
        mrange = list(range(lx, -1, -1)) + list(range(1, lx+1))
        mrange2 = list(range(-1*lx, 1)) + list(range(1, lx+1))
        scipy_p = lpmv(mrange, lx, np.cos(disp_sph[1]))

        #print('lx', lx, '-------------')

        for mxi, mx in enumerate(mrange2):

            re_exp = np.cos(mx*disp_sph[2])
            im_exp = np.sin(mx*disp_sph[2])

            #print('mx', mx, im_exp)

            val = math.sqrt(math.factorial(
                lx - abs(mx))/math.factorial(lx + abs(mx)))
            val *= scipy_p[mxi]

            irad = disp_sph[0] ** (lx)

            scipy_real = re_exp * val * irad
            scipy_imag = im_exp * val * irad

            ppmd_mom_re = moments[re_lm(lx, mx)]
            ppmd_mom_im = moments[im_lm(lx, mx)]

            phi_sph_re += scipy_real*ppmd_mom_re - scipy_imag*ppmd_mom_im
            phi_sph_im += scipy_real*ppmd_mom_im + ppmd_mom_re*scipy_imag

    return phi_sph_re, phi_sph_im


def phi_local(llimit, moments, disp_sph):
    """
    Compute the local expansion at a point from a charge
    """

    def re_lm(l,m): return (l**2) + l + m
    def im_lm(l,m): return (l**2) + l +  m + llimit**2

    for lx in range(llimit):
        mrange = list(range(lx, -1, -1)) + list(range(1, lx+1))
        mrange2 = list(range(-1*lx, 1)) + list(range(1, lx+1))
        scipy_p = lpmv(mrange, lx, np.cos(disp_sph[1]))

        #print('lx', lx, '-------------')

        for mxi, mx in enumerate(mrange2):

            re_exp = np.cos(-mx*disp_sph[2])
            im_exp = np.sin(-mx*disp_sph[2])

            #print('mx', mx, im_exp)

            val = math.sqrt(math.factorial(
                lx - abs(mx))/math.factorial(lx + abs(mx)))
            val *= scipy_p[mxi]

            irad = 1.0 / (disp_sph[0] ** (lx + 1))

            scipy_real = re_exp * val * irad
            scipy_imag = im_exp * val * irad

            moments[re_lm(lx, mx)] = scipy_real
            moments[im_lm(lx, mx)] = scipy_imag





def test_local_expansion_creation_evaluation():

    L = 20
    ncomp = 2 * (L**2)
    N = 50
    N2 = 10


    def re_lm(l,m): return (l**2) + l + m
    def im_lm(l,m): return (l**2) + l +  m + L**2

    
    lee = mc_expansion_tools.LocalExp(L)

    charges = np.zeros(N, REAL)
    radius = np.zeros(N, REAL)
    theta = np.zeros(N, REAL)
    phi = np.zeros(N, REAL)
    to_test = np.zeros((N, ncomp), REAL)
    ptr = np.zeros(N, ctypes.c_void_p)
    
    rng = np.random.RandomState(seed=892124357)
    
    charges[:] = rng.uniform(size=N)
    radius[:] = rng.uniform(1, 2, size=N)
    phi[:] = rng.uniform(0, 2*math.pi, size=N)
    theta[:] = rng.uniform(0, math.pi, N)


    for cx in range(N):
        ptr[cx] = to_test[cx, :].ctypes.get_as_parameter().value

    lee.compute_local_exp(
        N,
        charges,
        radius,
        theta,
        phi,
        ptr
    )



    #for cx in range(N):

    #    tmp = np.zeros(ncomp, REAL)
    #    sph = (radius[cx], theta[cx], phi[cx])
    #    phi_local(L, tmp, sph)
    #    
    #    tmp *= charges[cx]
    #    err = np.linalg.norm(tmp.ravel() - to_test[cx, :].ravel(), np.inf)
    #    assert err < 10.**-13



    # random evals

    assert N2 < N
    for cx in range(N):
        pr = np.array(rng.uniform(0.1, 2, N2), REAL)
        pt = np.array(rng.uniform(0, math.pi, N2), REAL)
        pp = np.array(rng.uniform(0, 2*math.pi, N2), REAL)
        pv = np.zeros(N2, REAL)

        vptr = np.zeros(N2, ctypes.c_void_p)

        for dx in range(N2):
            vptr[dx] = to_test[dx, :].ctypes.get_as_parameter().value

        lee.compute_phi_local(N2, pr, pt, pp, vptr, pv)
        
        for dx in range(N2):
            correct = compute_phi_local(L, to_test[dx, :], (pr[dx], pt[dx], pp[dx]))[0]
            err = abs(correct - pv[dx]) / abs(correct)
            assert err < 10.**-12





















