




import numpy as np
from ctypes import *
import sys
import time
import math

from ppmd import *

import ppmd.lib as lib


from coulomb_mc.mc_fmm_lm import MCFMM_LM
from coulomb_mc.mc_fmm_mm import MCFMM_MM
from coulomb_mc.mc_short_range import NonBondedDiff

PairLoop = pairloop.CellByCellOMP
ParticleLoop = loop.ParticleLoopOMP
State = state.State
PositionDat = data.PositionDat
ParticleDat = data.ParticleDat
ScalarArray = data.ScalarArray
Kernel = kernel.Kernel
GlobalArray = data.GlobalArray
Constant = kernel.Constant




class ParseDL_MONTE:
    def __init__(self, FIELD, CONTROL, atom_types=('Na core', 'Cl core')):
        
        get_con = utility.dl_poly.get_control_value
        get_fie = utility.dl_poly.get_field_value

        self.steps = int(get_con(CONTROL, 'STEPS')[0][0])
        self.cutoff = float(get_fie(FIELD, 'CUTOFF')[0][0])
        self.N = int(get_fie(FIELD, 'MAXATOM')[0][0])

        type0 = get_fie(FIELD, atom_types[0])
        self.mass0 = float(type0[0][0])
        self.charge0 = float(type0[0][1])
        type1 = get_fie(FIELD, atom_types[1])
        self.mass1 = float(type1[0][0])
        self.charge1 = float(type1[0][1])

        self.atoms_types_map ={
            atom_types[0]: 0,
            atom_types[1]: 1
        }

        self.mass_map = {
            atom_types[0]: self.mass0,
            atom_types[1]: self.mass1
        }

        self.charge_map = {
            atom_types[0]: self.charge0,
            atom_types[1]: self.charge1
        }

        lj00 = get_fie(FIELD, '{} {} LJ'.format(atom_types[0], atom_types[0]))
        lj00_eps = float(lj00[0][0])
        lj00_sigma = float(lj00[0][1])
        lj11 = get_fie(FIELD, '{} {} LJ'.format(atom_types[1], atom_types[1]))
        lj11_eps = float(lj11[0][0])
        lj11_sigma = float(lj11[0][1])

        lj01 = get_fie(FIELD, '{} {} LJ'.format(atom_types[0], atom_types[1]))
        if len(lj01) == 0:
            lj01 = get_fie(FIELD, '{} {} LJ'.format(atom_types[1], atom_types[0]))

        lj01_eps = float(lj01[0][0])
        lj01_sigma = float(lj01[0][1])

        
        cutoff = self.cutoff

        shift00 = (lj00_sigma/cutoff)**6. - (lj00_sigma/cutoff)**12.
        shift01 = (lj01_sigma/cutoff)**6. - (lj01_sigma/cutoff)**12.
        shift11 = (lj11_sigma/cutoff)**6. - (lj11_sigma/cutoff)**12.

        eoscoeff = -2.0 * lj01_eps/(lj01_sigma**2.0) + lj00_eps/(lj00_sigma**2.0) + \
            lj11_eps/(lj11_sigma**2.0)

        constants = {
            'e00'         : lj00_eps,
            'e01me00'     : lj01_eps - lj00_eps,
            'e_coeff'     : -2.*(lj01_eps - lj00_eps) + lj11_eps - lj00_eps,
            's00'         : lj00_sigma**2.0,
            's01ms00'     : lj01_sigma**2.0 - lj00_sigma**2.0,
            's_coeff'     : -2.*(lj01_sigma**2.0 - lj00_sigma**2.0) + lj11_sigma**2.0 - lj00_sigma**2.0,
            'eos00'       : lj00_eps/(lj00_sigma**2.0),
            'eos01meos00' : lj01_eps/(lj01_sigma**2.0) - lj00_eps/(lj00_sigma**2.0),
            'eos_coeff'   : eoscoeff,
            'ljs00'       : shift00,
            'ljs01mljs00' : shift01 - shift00,
            'ljs_coeff'   : -2.*(shift01 - shift00) + shift11 - shift00,
            'rc2'         : cutoff**2.
        }

        # 2 species kernel that should vectorise assuming types in {0, 1}

        header_src = r'''
        #include <math.h>
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

            // epsilon
            double e = {e00};
            e += dt0 * {e01me00};
            e += dt1 * {e01me00};
            e += dt0dt1 * {e_coeff};

            // sigma^2
            double s = {s00};
            s += dt0 * {s01ms00};
            s += dt1 * {s01ms00};
            s += dt0dt1 * {s_coeff};

            // potential shift
            double ljs = {ljs00};
            ljs += dt0 * {ljs01mljs00};
            ljs += dt1 * {ljs01mljs00};
            ljs += dt0dt1 * {ljs_coeff};

            // avoid the divide
            double eos = {eos00};
            eos += dt0 * {eos01meos00};
            eos += dt1 * {eos01meos00};
            eos += dt0dt1 * {eos_coeff};

            // compute the interaction
            const double r_m2 = s/r2;
            const double r_m4 = r_m2*r_m2;
            const double r_m6 = r_m4*r_m2;

            return (r2 < {rc2}) ? 2. * e * ((r_m6-1.0)*r_m6 + ljs) : 0.0;           
        }}
        '''.format(**constants)

        self.interaction_func = header_src

        header = lib.build.write_header(header_src)


        lj_kernel_code = '''
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

        PROP_DIFF[0] += u1 - u0;

        '''


        self.ljkernel = Kernel('mc_two_species_lj', lj_kernel_code, headers=(header,))




if __name__ == '__main__':


    dl_parser = ParseDL_MONTE('FIELD', 'CONTROL')
    
    rng = np.random.RandomState(1231)
    
    N1 = int(100000**(1./3.))
    e = 3.3 * N1
    E = (e, e, e)

    method = sys.argv[1]
    steps = dl_parser.steps
    max_hop_distance = E[0] * 0.1

    ikbT = 1.0


    if len(sys.argv) > 2:
        N1 = int(sys.argv[2])
        steps = int(sys.argv[3])
    N = N1**3


    R = int(max(3, math.log(N, 8)))
    L = 12


    print("-" * 80)
    print("N:", N)
    print("L:", L)
    print("R:", R)
    print("Method:", method)
    print("E:", E)
    print("Steps:", steps)
    print("-" * 80)


    # make a state object and set the global number of particles N
    A = State()

    # give the state a domain and boundary condition
    A.domain = domain.BaseDomainHalo(extent=E)
    A.domain.boundary_condition = domain.BoundaryTypePeriodic()

    # add a PositionDat to contain positions
    A.pos = PositionDat(ncomp=3)
    A.charge = ParticleDat(ncomp=1)
    A.types = ParticleDat(ncomp=1, dtype=c_int)


    # on MPI rank 0 add a cubic lattice of particles to the system 
    # with standard normal velocities
    rng = np.random.RandomState(512)

    initial_charges = rng.uniform(-10.0, 10.0, (N, 1))
    bias = np.sum(initial_charges) / N
    initial_charges -= bias   


    with A.modify() as AM:
        if A.domain.comm.rank == 0:
            AM.add({
                A.pos: utility.lattice.cubic_lattice((N1, N1, N1), E),
                A.charge: initial_charges,
            })
    

    if method == 'lm':
        MC = MCFMM_LM(A.pos, A.charge, A.domain, 'pbc', r=R, l=L)
    elif method == 'mm':
        MC = MCFMM_MM(A.pos, A.charge, A.domain, 'pbc', r=R, l=L)
    else:
        raise RuntimeError('Bad method chosen.')

    
    MC.initialise()
    MC_LOCAL = NonBondedDiff(A, A.types, 'pbc', dl_parser.cutoff, dl_parser.interaction_func)
    MC_LOCAL.initialise()

    accept_count = 0
    step_count = 0

    
    t0 = time.time()
    t0_outer = time.time()
    for stepx in range(steps):
        
        direction = rng.uniform(low=-1.0, high=1.0, size=3)
        magnitude = np.linalg.norm(direction)
        new_magnitude = rng.uniform(0, max_hop_distance)
        direction = direction * (new_magnitude /  magnitude)

        particle_id = rng.randint(N)

        prop_pos = A.pos[particle_id, :].copy() + direction

        for dx in (0, 1, 2):
            if prop_pos[dx] < E[dx] * -0.5: prop_pos[dx] += E[dx]
            elif prop_pos[dx] > E[dx] * 0.5: prop_pos[dx] -= E[dx]
        

        
        move = (particle_id, prop_pos)

        du_local = MC_LOCAL.propose(move)
        du_elec = MC.propose(move)
        
        du = du_local + du_elec

        if du < 0:
            MC_LOCAL.accept(move, du_local)
            MC.accept(move, du_elec)
            accept_count += 1
        elif math.exp(-du * ikbT) > rng.uniform():
            MC_LOCAL.accept(move, du_local)
            MC.accept(move, du_elec)
            accept_count += 1
        
        step_count += 1
        if accept_count > 0 and stepx % 1000 == 0:
            print((time.time() - t0) / accept_count, accept_count / step_count, MC.energy + MC_LOCAL.energy)
            t0 = time.time()
            accept_count = 0
            step_count = 0


    t1_outer = time.time()
    
    print("Time taken:", t1_outer - t0_outer)

    opt.print_profile()





















