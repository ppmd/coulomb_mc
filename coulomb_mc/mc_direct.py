

import numpy as np
from ppmd import data, loop, kernel, access, lib, opt
from ppmd.coulomb import fmm_interaction_lists, octal
from coulomb_kmc import kmc_expansion_tools, common
import ctypes
import math

from coulomb_mc import mc_expansion_tools

import time

from coulomb_mc.mc_common import MCCommon


Constant = kernel.Constant
REAL = ctypes.c_double
INT64 = ctypes.c_int64

PROFILE = opt.PROFILE




class DirectCommon(MCCommon):

    def __init__(self, positions, charges, domain, boundary_condition, r, subdivision, dat_cells, dat_gids):

        self.positions = positions
        self.charges = charges
        self.domain = domain
        self.comm = self.domain.comm
        self.boundary_condition = boundary_condition
        self.R = r
        self.subdivision = subdivision
        self.group = self.positions.group

        self.dat_cells = dat_cells
        self.dat_gids = dat_gids

        assert dat_gids.ncomp == 1
        assert dat_gids.dtype == INT64

        assert dat_cells.ncomp == 3
        assert dat_cells.dtype == INT64


        # interaction lists
        self.il = fmm_interaction_lists.compute_interaction_lists(domain.extent, self.subdivision)
        self.il_earray = np.array(self.il[1], INT64)


        s = self.subdivision
        s = [sx ** (self.R - 1) for sx in s]
        self.cell_occupation = np.zeros((s[2], s[1], s[0]), INT64)
        self.cell_indices = np.zeros((s[2], s[1], s[0], 10), INT64)
        self.max_occupancy = None

        self.direct_map = {}
        self._direct_contrib_lib = None
        self._init_direct_contrib_lib()


    def initialise(self):
        
        g = self.group
        N = g.npart_local

        self.direct_map = {}
        C = self.dat_cells
        for px in range(N):

            # cell on finest level
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

    
    def get_old_energy(self, px):

        C = self.dat_cells

        energy = 0.0

        energy_pc = REAL()
        index_c = INT64(px)
        
        t0 = time.time()
        self._direct_contrib_lib(
            INT64(0),                         #// -1 for "all old contribs (does not use id array)", 0 for indexed old contribs, 1 for new contribs
            INT64(1),                          #//number of contributions to compute
            INT64(self.il_earray.shape[0]),
            INT64(self.cell_indices.shape[3]),
            ctypes.byref(index_c),
            self.dat_gids.view.ctypes.get_as_parameter(),
            self.positions.view.ctypes.get_as_parameter(),
            self.charges.view.ctypes.get_as_parameter(),
            self.dat_cells.view.ctypes.get_as_parameter(),
            self.cell_occupation.ctypes.get_as_parameter(),
            self.cell_indices.ctypes.get_as_parameter(),
            self.il_earray.ctypes.get_as_parameter(),
            ctypes.byref(REAL()),
            ctypes.byref(INT64()),
            ctypes.byref(energy_pc)
        )
        energy_c = energy_pc.value
        energy += energy_c
        self._profile_inc('direct_old', time.time() - t0)

        return energy
    

    
    def get_new_energy(self, px, pos):

        energy = 0.0

        # cell on finest level
        cell = self._get_cell(pos)

        t0 = time.time()
        energy_pc = REAL(0)
        index_c = INT64(px)
        tpos = np.array((pos[0], pos[1], pos[2]), REAL)
        tcell = np.array(cell, INT64)
        self._direct_contrib_lib(
            INT64(1),                         #// -1 for "all old contribs (does not use id array)", 0 for indexed old contribs, 1 for new contribs
            INT64(1),                          #//number of contributions to compute
            INT64(self.il_earray.shape[0]),
            INT64(self.cell_indices.shape[3]),
            ctypes.byref(index_c),
            self.dat_gids.view.ctypes.get_as_parameter(),
            self.positions.view.ctypes.get_as_parameter(),
            self.charges.view.ctypes.get_as_parameter(),
            self.dat_cells.view.ctypes.get_as_parameter(),
            self.cell_occupation.ctypes.get_as_parameter(),
            self.cell_indices.ctypes.get_as_parameter(),
            self.il_earray.ctypes.get_as_parameter(),
            tpos.ctypes.get_as_parameter(),
            tcell.ctypes.get_as_parameter(),
            ctypes.byref(energy_pc)
        )
        energy_c = energy_pc.value
        energy += energy_c
        self._profile_inc('direct_new', time.time() - t0)

        return energy

    

    def accept(self, move):

        t0 = time.time()
        px = int(move[0])
        new_pos = move[1]
        
        g = self.positions.group
        old_cell = self.dat_cells[px, :].copy()
        new_cell = self._get_cell(new_pos)
        
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
        
        self._profile_inc('direct_accept', time.time() - t0)       


    def _make_occupancy_map(self):
        s = self.subdivision
        s = [sx ** (self.R - 1) for sx in s]
        if self.cell_indices.shape[3] < self.max_occupancy:
            self.cell_indices = np.zeros((s[2], s[1], s[0], self.max_occupancy), INT64)
        
        for cellx in self.direct_map.keys():
            l = len(self.direct_map[cellx])
            self.cell_indices[cellx[2], cellx[1], cellx[0], :l] = self.direct_map[cellx]
            self.cell_occupation[cellx[2], cellx[1], cellx[0]] = len(self.direct_map[cellx])


 
    def _init_direct_contrib_lib(self):

        source = r'''

        extern "C" int direct_entry(
            const INT64 MODE,                   // -1 for "all old contribs (does not use id array)", 0 for indexed old contribs, 1 for new contribs
            const INT64 n,                      //number of contributions to compute
            const INT64 noffsets,               //number of nearest neighbour cells
            const INT64 occ_stride,
            const INT64 * RESTRICT I,           //ids to compute (might be null)
            const INT64 * RESTRICT IDS,         //particle ids
            const REAL * RESTRICT P,            //positions
            const REAL * RESTRICT Q,            //charges
            const INT64 * RESTRICT CELLS,       //particle cells
            const INT64 * RESTRICT OCC,         //cell occupancies
            const INT64 * RESTRICT MAP,         //map from cells to particle ids (local ids)
            const INT64 * RESTRICT NNMAP,       //nearest neighbour offsets
            const REAL * RESTRICT NEW_POS,      //new position if MODE == 1
            const INT64 * RESTRICT NEW_CELL,    //new cells if MODE == 1
            REAL * RESTRICT U                   //output energy array
        ){{
            
            REAL UTOTAL = 0.0;

            #pragma omp parallel for reduction(+: UTOTAL) if((n>1))
            for(INT64 px=0 ; px<n ; px++){{
                
                INT64 ix1;
                if (MODE < 0) {{
                    ix1 = px;
                }} else {{
                    ix1 = I[px];
                }}


                const INT64 ix = ix1;
                REAL UTMP = 0.0;
 
                REAL rpx = P[ix * 3 + 0];
                REAL rpy = P[ix * 3 + 1];
                REAL rpz = P[ix * 3 + 2];

                if (MODE == 1){{
                    rpx = NEW_POS[px * 3 + 0];
                    rpy = NEW_POS[px * 3 + 1];
                    rpz = NEW_POS[px * 3 + 2];
                }}


                const REAL rx = rpx;
                const REAL ry = rpy;
                const REAL rz = rpz;

                INT64 CIX = CELLS[ix * 3 + 0];
                INT64 CIY = CELLS[ix * 3 + 1];
                INT64 CIZ = CELLS[ix * 3 + 2];

                if (MODE == 1){{
                    CIX = NEW_CELL[px * 3 + 0];
                    CIY = NEW_CELL[px * 3 + 1];
                    CIZ = NEW_CELL[px * 3 + 2];
                }}



                const REAL qi = Q[ix];
                const INT64 idi = IDS[ix];
                
                //for each offset


                //#pragma omp parallel for reduction(+: UTMP) if((n==1)) schedule(static,1)
                for(INT64 ox=0 ; ox<noffsets ; ox++){{
                    
                    const INT64 ocx = CIX + NNMAP[ox * 3 + 0];
                    const INT64 ocy = CIY + NNMAP[ox * 3 + 1];
                    const INT64 ocz = CIZ + NNMAP[ox * 3 + 2];

                    if (ocx < 0) {{continue;}}
                    if (ocy < 0) {{continue;}}
                    if (ocz < 0) {{continue;}}
                    if (ocx >= LCX) {{continue;}}
                    if (ocy >= LCY) {{continue;}}
                    if (ocz >= LCZ) {{continue;}}

                    // for each particle in the cell
                    
                    const INT64 cj = ocx + LCX * (ocy + LCY * ocz);
                    for(INT64 jxi=0 ; jxi<OCC[cj] ; jxi++){{

                        const INT64 jx = MAP[occ_stride * cj + jxi];

                        const REAL rjx = P[jx * 3 + 0];
                        const REAL rjy = P[jx * 3 + 1];
                        const REAL rjz = P[jx * 3 + 2];
                        const REAL qj = Q[jx];
                        const INT64 idj = IDS[jx];

                        const REAL dx = rx - rjx;
                        const REAL dy = ry - rjy;
                        const REAL dz = rz - rjz;

                        const REAL r2 = dx*dx + dy*dy + dz*dz;

                        const bool same_id = (MODE > 0) || ( (MODE < 1) && (idi != idj) );
                        UTMP += (same_id) ? qj / sqrt(r2) : 0.0;
                    
                    }}
                }}

                    
                if (MODE != -1) {{
                    U[px] = UTMP * qi;
                }} else {{
                    UTOTAL += UTMP * qi;
                }}
            }}

            if (MODE == -1){{
                U[0] = UTOTAL;
            }}

            return 0;
        }}
        '''.format(
        )

        header = r'''
        #include <math.h>
        #include <stdio.h>
        #define REAL double
        #define INT64 int64_t
        #define LCX {LCX}
        #define LCY {LCY}
        #define LCZ {LCZ}
        '''.format(
            LCX=self.subdivision[0] ** (self.R - 1),
            LCY=self.subdivision[1] ** (self.R - 1),
            LCZ=self.subdivision[2] ** (self.R - 1)
        )

        self._direct_contrib_lib = lib.build.simple_lib_creator(header, source)['direct_entry']







