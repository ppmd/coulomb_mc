
import numpy as np
from coulomb_mc import mc_fmm, mc_fmm_mm
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


    MC = mc_fmm_mm.MCFMM_MM(A.P, A.Q, A.domain, 'free_space', R, L)

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


    MC = mc_fmm_mm.MCFMM_MM(A.P, A.Q, A.domain, 'free_space', R, L)

    MC.initialise()
    
    FMM = coulomb.fmm.PyFMM(A.domain, r=R, l=L, free_space=True)

    FMM(A.P, A.Q, potential=A.U)


    for px in range(N):

        to_test = MC._get_old_energy(px)
        
        correct = A.U[px, 0]

        err = abs(to_test - correct) / abs(correct)
        
        assert err < 10.**-6





















