import pytest
import numpy as np
from coulomb_mc import mc_fmm, mc_fmm_lm, mc_fmm_mm
from ppmd import *

from coulomb_kmc import kmc_direct, kmc_fmm
import math

import ctypes

INT64 = ctypes.c_int64
REAL = ctypes.c_double

@pytest.mark.parametrize("MM_LM", (mc_fmm_lm.MCFMM_LM, mc_fmm_mm.MCFMM_MM))
def test_pbc_1(MM_LM):

    N = 100
    e = 10.
    R = 4
    L = 16


    rng = np.random.RandomState(3418)

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(e, e, e))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()


    A.P = data.PositionDat()
    A.Q = data.ParticleDat(ncomp=1)

    
    pi = np.array(rng.uniform(low=-0.5*e, high=0.5*e, size=(N, 3)), REAL)
    qi = np.array(rng.uniform(low=-1, high=1, size=(N, 1)), REAL)
    bias = np.sum(qi) / N
    qi -= bias


    with A.modify() as m:
        m.add({
            A.P: pi,
            A.Q: qi,
        })


    MC = MM_LM(A.P, A.Q, A.domain, 'pbc', R, L)

    MC.initialise()

    DFS = kmc_direct.PBCDirect(e, A.domain, L)

    correct = DFS(N, A.P.view, A.Q.view)

    err = abs(MC.energy - correct) / abs(correct)
    assert err < 10.**-6


@pytest.mark.parametrize("MM_LM", (mc_fmm_lm.MCFMM_LM, mc_fmm_mm.MCFMM_MM))
def test_pbc_2(MM_LM):

    N = 100
    e = 10.
    R = max(4, int(math.log(4*N, 8)))
    L = 12


    rng = np.random.RandomState(34118)

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(e, e, e))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()


    A.P = data.PositionDat()
    A.Q = data.ParticleDat(ncomp=1)
    A.G = data.ParticleDat(ncomp=1, dtype=INT64)

    
    pi = np.array(rng.uniform(low=-0.5*e, high=0.5*e, size=(N, 3)), REAL)
    qi = np.array(rng.uniform(low=-1, high=1, size=(N, 1)), REAL)
    bias = np.sum(qi) / N
    qi -= bias

    gi = np.arange(N).reshape((N, 1))

    with A.modify() as m:
        m.add({
            A.P: pi,
            A.Q: qi,
            A.G: gi
        })


    MC = MM_LM(A.P, A.Q, A.domain, 'pbc', R, L)
    MC.initialise()

    KMC = kmc_fmm.KMCFMM(A.P, A.Q, A.domain, boundary_condition='pbc', l=L, r=R)
    KMC.initialise()
    

    correct = KMC.energy
    err = abs(MC.energy - correct) / abs(correct)
    assert err < 10.**-5

    for testx in range(100):

        gid = rng.randint(0, N)
        lid = np.where(A.G.view[:, 0] == gid)[0]


        pos = rng.uniform(-0.5*e, 0.5*e, (3,))

        e0 = MC.propose((lid, pos.copy())) + MC.energy

        correct = KMC.propose(((lid, pos.copy()),))[0]


        
        err = abs(correct - e0) / abs(correct)
        assert err < 10.**-5

    
    KMC.free()
    MC.free()


@pytest.mark.parametrize("MM_LM", (mc_fmm_lm.MCFMM_LM, mc_fmm_mm.MCFMM_MM))
def test_pbc_3(MM_LM):

    N = 100
    e = 10.
    R = max(4, int(math.log(4*N, 8)))
    L = 16


    rng = np.random.RandomState(34118)

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(e, e, e))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()


    A.P = data.PositionDat()
    A.Q = data.ParticleDat(ncomp=1)
    A.G = data.ParticleDat(ncomp=1, dtype=INT64)

    
    pi = np.array(rng.uniform(low=-0.5*e, high=0.5*e, size=(N, 3)), REAL)
    qi = np.array(rng.uniform(low=-1, high=1, size=(N, 1)), REAL)
    bias = np.sum(qi) / N
    qi -= bias

    gi = np.arange(N).reshape((N, 1))

    with A.modify() as m:
        m.add({
            A.P: pi,
            A.Q: qi,
            A.G: gi
        })


    MC = MM_LM(A.P, A.Q, A.domain, 'pbc', R, L)
    
    MC.initialise()


    DFS = kmc_direct.PBCDirect(e, A.domain, L)

    correct = DFS(N, A.P.view, A.Q.view)

    err = abs(MC.energy - correct) / abs(correct)
    assert err < 10.**-5


    for testx in range(100):


        gid = rng.randint(0, N)
        lid = np.where(A.G.view[:, 0] == gid)[0]


        pos = rng.uniform(-0.5*e, 0.5*e, (3,))

        ed = MC.propose((lid, pos.copy()))
        MC.propose((lid, pos.copy()))
        e0 = ed + MC.energy

        old_pos = pi[gid, :].copy()


        #print("\t==>", get_old_energy(N, lid, pi, qi), get_new_energy(N, lid, pi, qi, pos), get_self_energy(qi[gid, 0], old_pos, pos))


        pi[gid, :] = pos.copy()
        correct = DFS(N, pi, qi)

        
        err = abs(correct - e0) / abs(correct)
        #print(err, correct, e0)

        assert err < 10.**-5
        

        # accept the move
        MC.accept((lid, pos.copy()), ed)


        assert abs(MC.energy - correct) < 10**-5


