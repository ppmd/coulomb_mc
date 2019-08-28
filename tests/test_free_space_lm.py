
import numpy as np
from coulomb_mc import mc_fmm, mc_fmm_lm
from ppmd import *

from coulomb_kmc import kmc_direct
import math

import ctypes

INT64 = ctypes.c_int64
REAL = ctypes.c_double

def test_free_space_1():

    N = 100
    e = 10.
    R = 4
    L = 12


    rng = np.random.RandomState(3418)

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(e, e, e))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()


    A.P = data.PositionDat()
    A.Q = data.ParticleDat(ncomp=1)

    
    pi = np.array(rng.uniform(low=-0.5*e, high=0.5*e, size=(N, 3)), REAL)
    qi = np.array(rng.uniform(low=-1, high=1, size=(N, 1)), REAL)

    with A.modify() as m:
        m.add({
            A.P: pi,
            A.Q: qi,
        })


    MC = mc_fmm_lm.MCFMM_LM(A.P, A.Q, A.domain, 'free_space', R, L)

    MC.initialise()

    DFS = kmc_direct.FreeSpaceDirect()

    correct = DFS(N, A.P.view, A.Q.view)

    err = abs(MC.energy - correct) / abs(correct)
    assert err < 10.**-6








def test_free_space_2():

    N = 100
    e = 10.
    R = 5
    L = 16


    rng = np.random.RandomState(3418)

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(e, e, e))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()


    A.P = data.PositionDat()
    A.Q = data.ParticleDat(ncomp=1)
    A.U = data.ParticleDat(ncomp=1)

    
    pi = np.array(rng.uniform(low=-0.5*e, high=0.5*e, size=(N, 3)), REAL)
    qi = np.array(rng.uniform(low=-1, high=1, size=(N, 1)), REAL)

    with A.modify() as m:
        m.add({
            A.P: pi,
            A.Q: qi,
        })


    MC = mc_fmm_lm.MCFMM_LM(A.P, A.Q, A.domain, 'free_space', R, L)

    MC.initialise()
    
    FMM = coulomb.fmm.PyFMM(A.domain, r=R, l=L, free_space=True)

    FMM(A.P, A.Q, potential=A.U)


    for px in range(N):

        to_test = MC._get_old_energy(px)
        
        correct = A.U[px, 0]

        err = abs(to_test - correct) / abs(correct)
        
        assert err < 3 * 10.**-6








def test_free_space_3():

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
    gi = np.arange(N).reshape((N, 1))

    with A.modify() as m:
        m.add({
            A.P: pi,
            A.Q: qi,
            A.G: gi
        })


    MC = mc_fmm_lm.MCFMM_LM(A.P, A.Q, A.domain, 'free_space', R, L)
    
    MC.initialise()



    DFS = kmc_direct.FreeSpaceDirect()

    correct = DFS(N, A.P.view, A.Q.view)

    err = abs(MC.energy - correct) / abs(correct)
    assert err < 10.**-5

    
    for testx in range(100):

        gid = rng.randint(0, N)
        lid = np.where(A.G.view[:, 0] == gid)[0]


        pos = rng.uniform(-0.5*e, 0.5*e, (3,))

        e0 = MC.propose((lid, pos.copy())) + MC.energy


        old_pos = pi[gid, :].copy()
        pi[gid, :] = pos.copy()

        correct = DFS(N, pi, qi)

        #te = 0.0
        #for px in range(N):
        #    te += get_old_energy(N, px, pi, qi)
        #te *= 0.5
        #print("te", te)


        pi[gid, :] = old_pos.copy()

        
        #print("\t==>", get_old_energy(N, lid, pi, qi), get_new_energy(N, lid, pi, qi, pos), get_self_energy(qi[gid, 0], old_pos, pos))
        
        err = abs(correct - e0) / abs(correct)
        #print(err, correct, e0)
        assert err < 10.**-5



def test_free_space_4():

    N = 100
    e = 10.
    R = max(4, int(math.log(4*N, 8)))
    R = 5
    L = 18


    rng = np.random.RandomState(34118)

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=(e, e, e))
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()


    A.P = data.PositionDat()
    A.Q = data.ParticleDat(ncomp=1)
    A.G = data.ParticleDat(ncomp=1, dtype=INT64)

    
    pi = np.array(rng.uniform(low=-0.5*e, high=0.5*e, size=(N, 3)), REAL)
    qi = np.array(rng.uniform(low=-1, high=1, size=(N, 1)), REAL)
    gi = np.arange(N).reshape((N, 1))

    with A.modify() as m:
        m.add({
            A.P: pi,
            A.Q: qi,
            A.G: gi
        })


    MC = mc_fmm_lm.MCFMM_LM(A.P, A.Q, A.domain, 'free_space', R, L)
    MCL = mc_fmm.MCFMM(A.P, A.Q, A.domain, 'free_space', R, L)
    
    MC.initialise()
    MCL.initialise()

    DFS = kmc_direct.FreeSpaceDirect()

    correct = DFS(N, A.P.view, A.Q.view)

    err = abs(MC.energy - correct) / abs(correct)
    assert err < 10.**-5


    for testx in range(100):


        gid = rng.randint(0, N)
        lid = np.where(A.G.view[:, 0] == gid)[0]


        pos = rng.uniform(-0.5*e, 0.5*e, (3,))

        ed = MC.propose((lid, pos.copy()))
        MCL.propose((lid, pos.copy()))
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
        MCL.accept((lid, pos.copy()), ed)


        assert abs(MC.energy - correct) < 10**-5














