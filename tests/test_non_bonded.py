import pytest
import numpy as np
from coulomb_mc import mc_fmm, mc_fmm_lm, mc_fmm_mm
from coulomb_mc.mc_short_range import *
from ppmd import *

from coulomb_kmc import kmc_direct, kmc_fmm
import math

import ctypes

INT64 = ctypes.c_int64
REAL = ctypes.c_double


PairLoop = pairloop.CellByCellOMP



class ShortRangeMC:
    def __init__(self, positions, types, ikernel, max_move, cutoff):
        
        self.propose_pos = data.ScalarArray(ncomp=3, dtype=REAL)
        self.propose_diff = data.GlobalArray(ncomp=1, dtype=REAL)
        
        self.d0 = data.GlobalArray(ncomp=1, dtype=REAL)
        self.d1 = data.GlobalArray(ncomp=1, dtype=REAL)


        self.propose_loop = PairLoop(
            ikernel,
            dat_dict={
                'P': positions(access.READ),
                'T': types(access.READ),
                'PROP_POS': self.propose_pos(access.READ),
                'PROP_DIFF': self.propose_diff(access.INC_ZERO),
                'D0': self.d0(access.INC_ZERO),
                'D1': self.d1(access.INC_ZERO),
            },
            shell_cutoff = (max_move + cutoff) * 1.01
        )

        #print(lib.build.LOADED_LIBS[-1])

    
    def propose(self, move):
        px, pos = move
        self.propose_pos[:] = pos
        self.propose_loop.execute(local_id=int(px))
        return self.propose_diff[0]




def _setup(cutoff):

    constants = {
        'rc2': cutoff*cutoff,
    }

    header_src = r'''
    #include <math.h>
    #include <stdio.h>
    static inline double POINT_EVAL(
        const double rix,
        const double riy,
        const double riz,
        const double rjx,
        const double rjy,
        const double rjz,
        const double dt0,
        const double dt1
    ){{
        const double R0 = rix - rjx;
        const double R1 = riy - rjy;
        const double R2 = riz - rjz;
        const double r2 = R0*R0 + R1*R1 + R2*R2;
        const double dt0dt1 = dt0*dt1;


        //if (r2 < {rc2}) {{
        //    printf("-----> %d %d %f %f %f\n", (int) dt0, (int) dt1, rjx, rjy, rjz);
        //}} else {{
        //    printf("       %d %d %f %f %f\n", (int) dt0, (int) dt1, rjx, rjy, rjz);
        //}}


        return (r2 < {rc2}) ? dt0dt1 * r2 : 0.0;
    }}
    '''.format(**constants)

    header = lib.build.write_header(header_src)

    
    kernel_code = r'''
    
    //printf("%d | %f %f %f\n", T.j[0], P.j[0], P.j[1], P.j[2]);

    const double u0 = POINT_EVAL(
        P.i[0],
        P.i[1],
        P.i[2],
        P.j[0],
        P.j[1],
        P.j[2],
        (double) T.i[0],
        (double) T.j[0]
    );

    const double u1 = POINT_EVAL(
        PROP_POS[0],
        PROP_POS[1],
        PROP_POS[2],
        P.j[0],
        P.j[1],
        P.j[2],
        (double) T.i[0],
        (double) T.j[0]
    );

    D0[0] += u0;
    D1[0] += u1;

    PROP_DIFF[0] += u1 - u0;
    '''


    ikernel = kernel.Kernel('mc_two_species_exp', kernel_code, headers=(header,))
    

    return header_src, ikernel






def test_non_bonded_1():

    N = 100
    E = (50., 40., 30)

    cutoff = 5.0
    max_move = 2.0



    rng = np.random.RandomState(3418)

    A = state.State()
    A.domain = domain.BaseDomainHalo(extent=E)
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()


    A.P = data.PositionDat()
    A.T = data.ParticleDat(ncomp=1, dtype=INT64)

    pi = np.zeros((N, 3), REAL)
    for dx in (0,1,2):
        pi[:, dx] = rng.uniform(low=-0.5*E[dx], high=0.5*E[dx], size=(N,))
    ti = np.arange(N).reshape(N,1)


    with A.modify() as m:
        m.add({
            A.P: pi,
            A.T: ti,
        })

    header_src, ikernel = _setup(cutoff)

    srmc = ShortRangeMC(A.P, A.T, ikernel, max_move, cutoff)
    nbd = NonBondedDiff(A, A.T, 'pbc', cutoff, header_src)
    nbd.initialise()

    
    for testx in range(10000):

        direction = rng.uniform(low=-1.0, high=1.0, size=3)
        magnitude = np.linalg.norm(direction)
        new_magnitude = rng.uniform(0, max_move)
        direction = direction * (new_magnitude /  magnitude)

        particle_id = rng.randint(N)

        prop_pos = A.P[particle_id, :].copy() + direction
        prop_pos_ppmd = prop_pos.copy()

        for dx in (0, 1, 2):
            if prop_pos[dx] < E[dx] * -0.5: prop_pos[dx] += E[dx]
            elif prop_pos[dx] > E[dx] * 0.5: prop_pos[dx] -= E[dx]
        
        
        move = (particle_id, prop_pos)
        move_ppmd = (particle_id, prop_pos_ppmd)
        
        sr1 = srmc.propose(move_ppmd)
        sr2 = nbd.propose(move)

        err = abs(sr1 - sr2) / abs(sr2) if abs(sr2) > 0.0 else abs(sr1 - sr2)

        
        assert err < 10.**-10

        nbd.accept(move)
        with A.P.modify_view() as mv:
            mv[particle_id, :] = prop_pos










