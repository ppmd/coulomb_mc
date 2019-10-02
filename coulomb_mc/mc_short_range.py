
import numpy as np
from ppmd import data, loop, kernel, access, lib, opt, pairloop
from ppmd.coulomb import fmm_interaction_lists, octal
from coulomb_kmc import kmc_expansion_tools, common
import ctypes
import math
from itertools import product

from coulomb_mc import mc_expansion_tools

import time

from coulomb_mc.mc_common import MCCommon, BCType


Constant = kernel.Constant
REAL = ctypes.c_double
INT64 = ctypes.c_int64

PROFILE = opt.PROFILE



class NonBondedDiff:
    def __init__(self, state, types, boundary_condition, cutoff, interaction_func):
        
        self.group = state
        self.domain = state.domain

        # dats
        self.positions = state.get_position_dat()
        self.types = types
        state._nbd_cells = data.ParticleDat(ncomp=3, dtype=INT64)
        self.cells = state._nbd_cells

        self.cutoff = cutoff
        self.boundary_condition = BCType(boundary_condition)
        self.sh = pairloop.state_handler.StateHandler(state=state, shell_cutoff=cutoff, pair=False)
        self.interaction_func = interaction_func

        assert self.boundary_condition == BCType.PBC
        
        
        s = [int(math.floor(ex / (cutoff*1.01))) for ex in self.domain.extent]
        for sz in s: assert sz > 0, "cutoff larger than domain extent"
        assert len(s) == 3


        self.s = s
        self.swidths = [ex / sx for ex, sx in zip(self.domain.extent, s)]

        self.cell_occupation = np.zeros((s[2], s[1], s[0]), INT64)
        self._ptr_cell_occupation = self.cell_occupation.ctypes.get_as_parameter()
        self.cell_indices = np.zeros((s[2], s[1], s[0], 10), INT64)
        self._ptr_cell_indices = self.cell_indices.ctypes.get_as_parameter()
        self.max_occupancy = None

        self.direct_map = {}
        
        self.offset_map = np.zeros((27, 3), INT64)
        o = (-1, 0, 1)
        self.offset_map[:] = tuple(product(o,o,o))


        self.lib = self._generate_lib()

    def accept(self, move):
        px = int(move[0])
        new_pos = move[1]
        
        g = self.group
        old_cell = self.cells[px, :].copy()
        new_cell = self._new_cell_bin(new_pos)
 
        with self.cells.modify_view() as mv:
            mv[px, :] = new_cell

        # correct the cell to particle maps
        self.direct_map[(old_cell[0], old_cell[1], old_cell[2])].remove(px)
        assert px not in self.direct_map[(old_cell[0], old_cell[1], old_cell[2])]

        tnew = (new_cell[0], new_cell[1], new_cell[2])
        told = (old_cell[0], old_cell[1], old_cell[2])

        if tnew in self.direct_map.keys():
            self.direct_map[tnew].append(px)
        else:
            self.direct_map[tnew] = [px]
        
        possible_new_max = len(self.direct_map[tnew])
        ol = len(self.direct_map[told])


        if possible_new_max > self.max_occupancy:
            # need to remake the map
            self.max_occupancy = possible_new_max
            self._make_occupancy_map()
        else:
            # add the new one
            self.cell_indices[tnew[2], tnew[1], tnew[0], :possible_new_max] = self.direct_map[tnew]
            # remove the old one
            self.cell_indices[told[2], told[1], told[0], :ol] = self.direct_map[told]

        self.cell_occupation[tnew[2], tnew[1], tnew[0]] = possible_new_max
        self.cell_occupation[told[2], told[1], told[0]] = ol


    
    def _cell_bin(self):
        N = self.cells.npart_local
        P = self.positions.view
        with self.cells.modify_view() as mv:
            for dx in (0, 1, 2):
                mv[:, dx] = (P[:, dx] + 0.5*self.domain.extent[dx])/ self.swidths[dx]
                assert np.min(mv[:, dx]) > -1
                assert np.max(mv[:, dx]) < self.s[dx]


    def initialise(self):
        g = self.group
        N = g.npart_local

        self.direct_map = {}

        self._cell_bin()
        C = self.cells.view

        for px in range(N):

            cell = tuple(C[px, :])

            if cell in self.direct_map.keys():
                self.direct_map[cell].append(px)
            else:
                self.direct_map[cell] = [px]
        
        o = 0
        for kx in self.direct_map.values():
            o = max(o, len(kx))
        self.max_occupancy = o

        self._make_occupancy_map()


    def _make_occupancy_map(self):
        s = self.s
        if self.cell_indices.shape[3] < self.max_occupancy:
            self.cell_indices = np.zeros((s[2], s[1], s[0], self.max_occupancy), INT64)
            self._ptr_cell_indices = self.cell_indices.ctypes.get_as_parameter()
        
        for cellx in self.direct_map.keys():
            l = len(self.direct_map[cellx])
            self.cell_indices[cellx[2], cellx[1], cellx[0], :l] = self.direct_map[cellx]
            self.cell_occupation[cellx[2], cellx[1], cellx[0]] = len(self.direct_map[cellx])
    
    def _new_cell_bin(self, pos):
        c = [int((pos[i] + 0.5*self.domain.extent[i])/self.swidths[i]) for i in (0,1,2)]
        return c



    def propose(self, move):
        px, pos = move
        eval_pos = np.array((pos[0], pos[1], pos[2]), REAL)
        eval_cell = np.array(self._new_cell_bin(pos), INT64)
        
        uold = REAL(0.0)
        unew = REAL(0.0)
        self.lib(
            INT64(0),
            INT64(px),
            eval_pos.ctypes.get_as_parameter(),
            eval_cell.ctypes.get_as_parameter(),
            self.offset_map.ctypes.get_as_parameter(),
            INT64(self.cell_indices.shape[3]),
            self.cell_occupation.ctypes.get_as_parameter(),
            self.cell_indices.ctypes.get_as_parameter(),
            self.sh.get_pointer(self.positions(access.READ)),
            self.sh.get_pointer(self.cells(access.READ)),
            self.sh.get_pointer(self.types(access.READ)),
            ctypes.byref(uold)
        )
        self.lib(
            INT64(1),
            INT64(px),
            eval_pos.ctypes.get_as_parameter(),
            eval_cell.ctypes.get_as_parameter(),
            self.offset_map.ctypes.get_as_parameter(),
            INT64(self.cell_indices.shape[3]),
            self.cell_occupation.ctypes.get_as_parameter(),
            self.cell_indices.ctypes.get_as_parameter(),
            self.sh.get_pointer(self.positions(access.READ)),
            self.sh.get_pointer(self.cells(access.READ)),
            self.sh.get_pointer(self.types(access.READ)),
            ctypes.byref(unew)
        )       


        #print(unew.value, uold.value)
        return unew.value - uold.value



    def _generate_lib(self):


        source = r"""
        
        extern "C" int non_bonded(
            const INT64            mode,                // 0 for old position, 1 for new
            const INT64            particle_id,
            const REAL  * RESTRICT eval_pos,            // used for both modes
            const INT64 * RESTRICT eval_cell,
            const INT64 * RESTRICT cell_offset_map,
            const INT64            occ_stride,
            const INT64 * RESTRICT OCC,                 // cell occupancies
            const INT64 * RESTRICT MAP,                 // map from cells to particle ids (local ids)
            const REAL  * RESTRICT dat_pos,
            const INT64 * RESTRICT dat_cell,
            const INT64 * RESTRICT dat_type,
                  REAL  * RESTRICT U                    // output energy
        ){{
            
            const REAL rix  = (mode) ? eval_pos[0]  : dat_pos[particle_id * 3 + 0];
            const REAL riy  = (mode) ? eval_pos[1]  : dat_pos[particle_id * 3 + 1];
            const REAL riz  = (mode) ? eval_pos[2]  : dat_pos[particle_id * 3 + 2];
            const INT64 cix = (mode) ? eval_cell[0] : dat_cell[particle_id * 3 + 0];
            const INT64 ciy = (mode) ? eval_cell[1] : dat_cell[particle_id * 3 + 1];
            const INT64 ciz = (mode) ? eval_cell[2] : dat_cell[particle_id * 3 + 2];
            const INT64 ti  = dat_type[particle_id];

            REAL energy = 0.0;

            // for eacl cell offset
            for(int ox=0 ; ox<N_CELL_OFFSETS ; ox++){{
                
                INT64 cjx = cix + cell_offset_map[ox * 3 + 0];
                INT64 cjy = ciy + cell_offset_map[ox * 3 + 1];
                INT64 cjz = ciz + cell_offset_map[ox * 3 + 2];
                
                const REAL pbcx = (cjx < 0) ? -1.0 * EX : ( (cjx >= SX) ? EX : 0.0 ); 
                const REAL pbcy = (cjy < 0) ? -1.0 * EY : ( (cjy >= SY) ? EY : 0.0 ); 
                const REAL pbcz = (cjz < 0) ? -1.0 * EZ : ( (cjz >= SZ) ? EZ : 0.0 ); 

                // PBC mapping of cells
                cjx = (cjx + 128*SX) % SX;
                cjy = (cjy + 128*SY) % SY;
                cjz = (cjz + 128*SZ) % SZ;

                const INT64 cj = cjx + SX*(cjy + SY*cjz);

                // for particle in cell
                for(INT64 jxi=0 ; jxi<OCC[cj] ; jxi++){{
                    const INT64 jx = MAP[occ_stride * cj + jxi];
                    
                    const REAL rjx = dat_pos[jx * 3 + 0];
                    const REAL rjy = dat_pos[jx * 3 + 1];
                    const REAL rjz = dat_pos[jx * 3 + 2];


                    const bool mask = (jx == particle_id) ? 0 : 1;
                    energy += (mask) ? POINT_EVAL(
                        rix,
                        riy,
                        riz,
                        (rjx + pbcx),
                        (rjy + pbcy),
                        (rjz + pbcz),
                        (double) ti,
                        (double) dat_type[jx]
                    ) : 0.0;

            
                }}

            }}

            U[0] = energy;
            
            return 0;
        }}
        """

        header = r"""
        #define REAL double
        #define INT64 int64_t
        #define SX {SX}
        #define SY {SY}
        #define SZ {SZ}
        #define EX {EX}
        #define EY {EY}
        #define EZ {EZ}
        #define N_CELL_OFFSETS 27

        {INTERACTION_FUNC}
        """.format(
            SX=self.s[0],
            SY=self.s[1],
            SZ=self.s[2],
            EX=self.domain.extent[0],
            EY=self.domain.extent[1],
            EZ=self.domain.extent[2],
            INTERACTION_FUNC=self.interaction_func
        )


        gen_lib = lib.build.simple_lib_creator(header, source)['non_bonded']


        return gen_lib















