


import os
import json
import numpy as np
from ctypes import *
import sys
import time
import math

from ppmd import *

import ppmd.coulomb.ewald_half
import ppmd.coulomb.fmm

import ppmd.lib as lib


from coulomb_mc.mc_fmm_lm import MCFMM_LM
from coulomb_mc.mc_fmm_mm import MCFMM_MM
from coulomb_mc.mc_short_range import NonBondedDiff

from scipy import constants as spc

PairLoop = pairloop.CellByCellOMP
ParticleLoop = loop.ParticleLoopOMP
State = state.State
PositionDat = data.PositionDat
ParticleDat = data.ParticleDat
ScalarArray = data.ScalarArray
Kernel = kernel.Kernel
GlobalArray = data.GlobalArray
Constant = kernel.Constant





class CheckPositions:

    def __init__(self, S):
        self.S = S
        self.OGA = GlobalArray(ncomp=1, dtype=c_int64)

        k = kernel.Kernel(
            'overlap_test',
            '''
            const double r0 = P.j[0] - P.i[0];
            const double r1 = P.j[1] - P.i[1];
            const double r2 = P.j[2] - P.i[2];
            const double rr = r0*r0 + r1*r1 + r2*r2;

            if (rr < 0.0000000001) {
                OGA[0]++;
            }
            '''
        )

        self.pl = PairLoop(
            k,
            {
                'P': self.S.pos(access.READ),
                'OGA': self.OGA(access.INC_ZERO)
            },
            0.10
        )


    def __call__(self, P):

        e = self.S.domain.extent

        for dx in (0, 1, 2):
            assert np.min(P[:, dx]) >= -0.5 * e[dx]
            assert np.max(P[:, dx]) <=  0.5 * e[dx]

        self.pl.execute()
        
        assert self.OGA[0] == 0





class ParseDL_MONTE:
    def __init__(self, FIELD, CONTROL, CONFIG, atom_types=('Na core', 'Cl core')):

        self.extent = utility.dl_monte.read_domain_extent(CONFIG)

        
        get_con = utility.dl_monte.get_control_value
        get_fie = utility.dl_monte.get_field_value

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

        positions, types = utility.dl_monte.read_positions(CONFIG)

        self.positions = positions
        
        #for dx in (0, 1, 2):
        #    self.positions[:, dx] -= 0.5 * self.extent[dx]

        self.charges = np.zeros((self.positions.shape[0], 1), c_double)
        self.types = np.zeros((self.positions.shape[0], 1), c_int64)

        for px in range(self.positions.shape[0]):
            self.charges[px, 0] = self.charge_map[types[px]]
            self.types[px, 0] = self.atoms_types_map[types[px]]



        lj00 = get_fie(FIELD, '{} {} SLJ'.format(atom_types[0], atom_types[0]))
        lj00_eps = float(lj00[0][0])
        lj00_sigma = float(lj00[0][1])
        lj11 = get_fie(FIELD, '{} {} SLJ'.format(atom_types[1], atom_types[1]))
        lj11_eps = float(lj11[0][0])
        lj11_sigma = float(lj11[0][1])

        lj01 = get_fie(FIELD, '{} {} SLJ'.format(atom_types[0], atom_types[1]))
        if len(lj01) == 0:
            lj01 = get_fie(FIELD, '{} {} SLJ'.format(atom_types[1], atom_types[0]))

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
            //double ljs = 0.0;
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




def main(L=12):
    seed = np.random.randint(2**32 - 1)
    rng = np.random.RandomState(seed)
    method = sys.argv[1]

    dl_parser = ParseDL_MONTE('FIELD', 'CONTROL', 'CONFIG')

    if len(sys.argv) > 2:
        USE_DL_MONTE = False
        N1 = int(sys.argv[2])
        steps = int(sys.argv[3])
        N1 = int(100000**(1./3.))
        e = 3.3 * N1
        E = (e, e, e)
        N = N1**3
        pi = utility.lattice.cubic_lattice((N1, N1, N1), E)
        initial_charges = rng.uniform(-10.0, 10.0, (N, 1))
        bias = np.sum(initial_charges) / N
        initial_charges -= bias
        ti = np.zeros((N, 1), c_int64)

    else:
        USE_DL_MONTE = True
        steps = dl_parser.steps
        E = dl_parser.extent

        pi = dl_parser.positions
        initial_charges = dl_parser.charges
        ti = dl_parser.types

        N = pi.shape[0]
    



    assert np.sum(initial_charges) < 10.**-8


    max_hop_distance = 1.0
    current_hop_distance = 0.8

    desired_accept_ratio = 0.37

    


    internal_to_ev = ppmd.coulomb.fmm.internal_to_ev()
    temperature = 273.0
    ikbT = (-1.0) / ((spc.Boltzmann / spc.elementary_charge) * temperature)




    R = int(max(3, math.log(N, 8)))


    print("-" * 80)
    print("cutoff:", dl_parser.cutoff)
    print("seed:", seed)
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
    A.types = ParticleDat(ncomp=1, dtype=c_int64)
    A.force = ParticleDat(ncomp=3, dtype=c_double)


    # on MPI rank 0 add a cubic lattice of particles to the system 
    # with standard normal velocities


    with A.modify() as AM:
        if A.domain.comm.rank == 0:
            AM.add({
                A.pos: pi,
                A.charge: initial_charges,
                A.types: ti
            })

    #CheckPositions(A)(pi)
    

    if method == 'lm':
        MC = MCFMM_LM(A.pos, A.charge, A.domain, 'pbc', r=R, l=L)
    elif method == 'mm':
        MC = MCFMM_MM(A.pos, A.charge, A.domain, 'pbc', r=R, l=L)
    else:
        raise RuntimeError('Bad method chosen.')

    
    MC.initialise()
    MC_LOCAL = NonBondedDiff(A, A.types, 'pbc', dl_parser.cutoff, dl_parser.interaction_func)
    MC_LOCAL.initialise()
    

    DOUBLE_CHECK = False
    
    if DOUBLE_CHECK:
        from ppmd.coulomb.direct import PBCDirect
        DIRECT = PBCDirect(E[0], A.domain, 16)
        direct_energy = DIRECT(N, A.pos.view.copy(), A.charge.view.copy())
        print("DIRECT:", direct_energy)
        EWALD = coulomb.ewald_half.EwaldOrthogonalHalf(A.domain, real_cutoff=dl_parser.cutoff)
        ewald_energy = EWALD(A.pos, A.charge, A.force)
        print("EWALD:", ewald_energy)
        print("EWALD REAL:", EWALD.last_real_energy * coulomb.fmm.internal_to_ev())
        print("EWALD RECIP:", EWALD.last_recip_energy * coulomb.fmm.internal_to_ev())
        print("EWALD SELF:", EWALD.last_self_energy * coulomb.fmm.internal_to_ev())


    with open(
            method + '_step_0_{L}.json'.format(
                L=L
            ),
            'w'
        ) as fh:
        fh.write(json.dumps({
            'method': method,
            'L': L,
            'energy_electrostatic': MC.energy,
            'energy_electrostatic_ev': MC.energy * coulomb.fmm.internal_to_ev(),
            'energy_vdw': MC_LOCAL.energy
        }))
    
    print("Initial Energy:")
    print("ES:", MC.energy * coulomb.fmm.internal_to_ev())
    print("NB:", MC_LOCAL.energy)

    print("-" * 80)

    accept_count = 0
    local_accept_count = 0
    local_step_count = 0
    
    ALWAYS_ACCEPT = False
    EXTRA_CHECKS = True
    
    t0 = time.time()
    t0_outer = time.time()
    for stepx in range(steps):
        
        direction = rng.uniform(low=-1.0, high=1.0, size=3)
        magnitude = np.linalg.norm(direction)
        new_magnitude = rng.uniform(0, current_hop_distance)
        direction = direction * (new_magnitude /  magnitude)

        particle_id = rng.randint(N)

        prop_pos = A.pos[particle_id, :].copy() + direction

        for dx in (0, 1, 2):
            if prop_pos[dx] < E[dx] * -0.5: prop_pos[dx] += E[dx]
            elif prop_pos[dx] > E[dx] * 0.5: prop_pos[dx] -= E[dx]

            assert prop_pos[dx] >= E[dx] * -0.5
            assert prop_pos[dx] <= E[dx] *  0.5
        

        
        move = (particle_id, prop_pos)

        du_local = MC_LOCAL.propose(move)
        du_elec = MC.propose(move)
        
        du = du_local + du_elec * internal_to_ev
        old_elec_energy = MC.energy
        old_local_energy = MC_LOCAL.energy

        
        if du < 0 or ALWAYS_ACCEPT:
            MC_LOCAL.accept(move, du_local)
            MC.accept(move, du_elec)
            accept_count += 1
            local_accept_count += 1
            MOVE_ACCEPTED = True

        elif math.exp(du * ikbT) > rng.uniform():
            MC_LOCAL.accept(move, du_local)
            MC.accept(move, du_elec)
            accept_count += 1
            local_accept_count += 1
            MOVE_ACCEPTED = True

        else:
            MOVE_ACCEPTED = False


        if MOVE_ACCEPTED and EXTRA_CHECKS:
            new_elec_energy = MC.energy
            err_elec = min(abs(old_elec_energy + du_elec - new_elec_energy) / abs(new_elec_energy), abs(old_elec_energy + du_elec - new_elec_energy))

            assert err_elec < 10.**-4, err_elec

            new_local_energy = MC_LOCAL.energy
            err_local = min(abs(old_local_energy + du_local - new_local_energy) / abs(new_local_energy), abs(old_local_energy + du_local - new_local_energy))
            
            assert err_local < 10.**-4, err_local



        
        local_step_count += 1

        local_accept_ratio = local_accept_count / local_step_count
        if accept_count > 0 and stepx % 100 == 0:
            if local_accept_ratio > 0:
                r = desired_accept_ratio / local_accept_ratio 
                h2 = current_hop_distance / r

                current_hop_distance += 0.25 * (h2 - current_hop_distance)
                current_hop_distance = min(current_hop_distance, max_hop_distance)


                local_step_count = 0
                local_accept_count = 0

            if EXTRA_CHECKS:
                print(MC.energy * internal_to_ev + MC_LOCAL.energy, current_hop_distance, local_accept_ratio, r)

            
    
         




    t1_outer = time.time()
    
    print("Time taken:", t1_outer - t0_outer)
    print("Overall Accept Ratio:", accept_count/steps)

    with open('timing_data_{METHOD}_{L}.json'.format(
        METHOD=method,
        L=L
    ), 'w') as fh:
        fh.write(json.dumps({
            'method': method,
            'L': L,
            'R': R,
            'N': N,
            'Nsteps' : steps,
            'time_taken': t1_outer - t0_outer,
            'accept_count': accept_count,
        }))

    opt.print_profile()




if __name__ == '__main__':
    L_set_file = 'L_set.json'
    if os.path.exists(L_set_file):
        with open(L_set_file) as fh:
            L_data = json.loads(fh.read())
        L_set = L_data['L_set']
        for Lx in L_set:
            main(L=Lx)
    else:
        main()

