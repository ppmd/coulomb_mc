import numpy as np
from ctypes import *
from ppmd import *

from coulomb_mc.mc_fmm_lm import MCFMM_LM

State = state.State
PositionDat = data.PositionDat
ParticleDat = data.ParticleDat

if __name__ == '__main__':

    # Number of charges
    N = 100
    # Extents of domain
    E = 10.
    # Iteration count
    Nsteps = 50000

    # Create a PPMD State and domain with ParticleDats for positions, charge and global id.
    S = State(
        domain=domain.BaseDomainHalo(
            extent=(E,E,E), 
            boundary_condition=domain.BoundaryTypePeriodic()
        ),
        particle_dats={
            'P' : PositionDat(),
            'Q' : ParticleDat(ncomp=1, dtype=c_double),
            'GID': ParticleDat(ncomp=1, dtype=c_int),
        }
    )
    
    # Add unit charges to the domain with uniform random positions.
    rng = np.random.RandomState(seed=123)

    with S.modify() as sm:
        sm.add(
            {
                S.P: rng.uniform(low=-0.5*E, high=0.5*E, size=(N,3)),
                S.Q: np.ones(shape=(N,1)),
                S.GID: np.arange(N).reshape((N,1))
            }
        )
    
    # We can now initialise the electrostatic solver. If we wanted periodic boundary
    # conditions we would use 'pbc' as the "boundary_condition" argument.
    # The "r" parameter determines the number of levels used in the gird hierarchy.
    # The "l" parameter determines the number of expansion terms used for the spherical
    # harmonic expansions. 
    # Please refer to the paper for discussion of the "r" and "l" parameters.
    solver = MCFMM_LM(
        positions=S.P, 
        charges=S.Q, 
        domain=S.domain,
        boundary_condition='free_space',
        r=3,
        l=12
    )
    
    # Initialise the electrostatic solver.
    solver.initialise()

    # We can now enter our main simulation loop where we propose and accept/reject moves.
    for ix in range(Nsteps):
        
        # randomly select a particle, N.B. the propose interface takes a particle id
        # and a proposed position inside the domain. The particle id is the index of
        # the particle in the ParticleDat.
        particle_id = rng.randint(N)
        offset = rng.uniform(low=-0.01*E, high=0.01*E, size=(3,))
        
        proposed_position = S.P[particle_id, :] + offset
        # map back into the simulation domain
        for dx in (0, 1, 2):
            if proposed_position[dx] < -0.5*E: proposed_position[dx] += E
            if proposed_position[dx] >  0.5*E: proposed_position[dx] -= E

        
        # the move is defined by the following tuple
        move = (particle_id, proposed_position)

        # get the electrostatic energy difference of the proposed move
        energy_change = solver.propose(move)

        # crude accept/reject determination.
        # The solver tracks the energy of the system (as a current implementation detail)
        # If the energy change is passed as an argument the energy difference will not be
        # recomputed. Future implementation versions may not need this requirement. 
        if energy_change < 0.0:
            solver.accept(move, energy_change)

        if ix % 1000 == 0:
            print(ix, move, energy_change, solver.energy)

    # Quantify the error in the final configuration energy
    from ppmd.coulomb.direct import FreeSpaceDirect
    direct_solver = FreeSpaceDirect()
    final_direct_energy = direct_solver(N, S.P.view, S.Q.view)
    print(abs(final_direct_energy - solver.energy))

