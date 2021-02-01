Introduction
============

FMM-MC, the algorithm this package implements, is a method to compute differences in electrostatic potential energies in Metropolis Hastings style Monte Carlo. More specifically, there is a class of simulation techniques where the state of the simulation is a set of charged particles that interact via a Coulomb potential. To advance the simulation a charge is selected at random and a new position is randomly sampled for this charge (often by adding a random offset vector). This new position is called the proposed position. Based on the electrostatic energy difference between the original position and the proposed position the move is either accepted or rejected. We support free-space and fully periodic boundary conditions.

This implementation provides methods to 1) initialise a FMM-MC electrostatic solver, 2) compute the difference in electrostatic energy between the original position and the proposed positions and 3) accept a charge move to the proposed position. For details of the underlying algorithm we refer the reader to the paper *A new algorithm for electrostatic interactions in Monte Carlo simulations of charged particles*. We now provide details of how to structure a program that uses PPMD to store particle data and use this package for electrostatic interactions.

We import helper modules ``numpy`` and ``ctypes`` before importing ``ppmd`` and the solver itself from ``coulomb_mc``,
::
    
    import numpy as np
    from ctypes import *

    from ppmd import *
    from coulomb_mc.mc_fmm_lm import MCFMM_LM


We then define constants that determine which types to use from PPMD,
::

    State = state.State
    PositionDat = data.PositionDat
    ParticleDat = data.ParticleDat

After the imports we can start to define our simulation. We shall create a cubic domain of side length ``E`` containing ``N`` positively charged particles. The boundary condition of the PPMD simulation domain determines how particles should be treated at the edge of the domain and does not influence the electrostatic solver, we choose the ``BoundaryTypePeriodic`` boundary condition. Later, for our electrostatic solver we shall choose the ``free_space`` boundary condition. In the following code snippet we create a PPMD state, which groups our particle data types together, and define the simulation domain and corresponding boundary condition:
::

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

Note that the names of the particle data structures are user choice. These properties will be accessible as attributes of the parent ``State`` instance. For example to access the particle global ids we would use the ``S.GID`` attribute.

Now that the particle data types have been defined we can add particles to our simulation domain. Particles are added or removed from the domain by using the ``modify`` method of the ``State`` class in a Python context. 
::

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

Now that the simulation domain contains particles that have positions and charge values we can define and initialise the electrostatic solver. Here we use the free space boundary conditions as an example. If we were to use the fully periodic boundary conditions we would have to ensure that our simulation did not contain a net charge. For an example simulation using the fully periodic boundary conditions please refer to the additional examples.

The solver is initialised with the ``PositionDat`` that contains the particle positions, the ``ParticleDat`` that contains the particle charge values, the simulation domain ``S.domain``, boundary condition, number of levels in the gird hierarchy ``r`` and the number of expansion terms ``l``:
::

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

For more information and guidance on how to choose the number of levels and expansion terms please see the corresponding publication. Based on our experiment we would expect approximately 5 significant figures of accuracy from 12 expansion terms. The number of levels should be chosen as :math:`\log_8 ( \alpha N)`. Where :math:`\alpha` is a machine dependent coefficient (we used :math:`\alpha \approx 0.327` on Ivy Bridge E2650v2).

We are now ready to enter our main simulation loop. For this introductory example we shall propose moves with a uniform random offset. Note that proposed moves must be in the simulation domain which is why we periodically wrap the proposed position. The proposed move is described to the electrostatic solver by providing a particle id (i.e. the integer row index in the ``ParticleDats``) and the proposed new position. In the introductory example we accept moves if the energy change is negative. It is important to note that when the accept method is called with a proposed move the electrostatic solver updates internal data structures **and** updates the particles position in the ``PositionDat``. In this case the ``PositionDat`` is stored as the attribute ``S.P`` which was passed into the solver constructor.
::
    
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

A full copy of this example is available :download:`here <../../examples/free_space_example/free_space.py>`.

