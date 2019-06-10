
import numpy as np
from coulomb_mc import mc_fmm
from ppmd import *

from coulomb_kmc import kmc_direct



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

    
    pi = rng.uniform(low=-0.5*e, high=0.5*e, size=(N, 3))
    qi = rng.uniform(low=-1, high=1, size=(N, 1))

    with A.modify() as m:
        m.add({
            A.P: pi,
            A.Q: qi
        })


    MC = mc_fmm.MCFMM(A.P, A.Q, A.domain, 'free_space', R, L)
    
    MC.initialise()

    DFS = kmc_direct.FreeSpaceDirect()

    correct = DFS(N, A.P.view, A.Q.view)

    err = abs(MC.energy - correct) / abs(correct)
    assert err < 10.**-6
