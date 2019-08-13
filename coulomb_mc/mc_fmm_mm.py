

import numpy as np
from ppmd import data, loop, kernel, access, lib, opt
from ppmd.coulomb import fmm_interaction_lists, octal, mm
from coulomb_kmc import kmc_expansion_tools, common
import ctypes
import math

from coulomb_mc import mc_expansion_tools
import time


from ppmd.coulomb.sph_harm import *

Constant = kernel.Constant
REAL = ctypes.c_double
INT64 = ctypes.c_int64

PROFILE = opt.PROFILE

class MCFMM_MM:

    def __init__(self, positions, charges, domain, boundary_condition, r, l):

        self.positions = positions
        self.charges = charges
        self.domain = domain
        self.comm = self.domain.comm
        self.boundary_condition = boundary_condition
        self.R = r
        self.L = l
        self.ncomp = (self.L ** 2) * 2
        self.group = self.positions.group


        self.mm = mm.PyMM(positions, charges, domain, boundary_condition, r, l)

        self.energy = None

        self._init_libs()


    def initialise(self):
        
        self.energy = self.mm(self.positions, self.charges)







    def propose(self, move):
        px = int(move[0])
        pos = move[1]

        old_energy = self._get_old_energy(px)
        print(old_energy)




    

    
    def _get_old_energy(self, ix):
        
        ie = REAL(0)
            
        self._old_indirect_lib(
            INT64(ix),
            self.mm.sh.get_pointer(self.positions(access.READ)),
            self.mm.sh.get_pointer(self.charges(access.READ)),
            self.mm.il_scalararray.ctypes_data,
            self.mm.sh.get_pointer(self.group._mm_cells(access.READ)),
            self.mm.sh.get_pointer(self.group._mm_child_index(access.READ)),
            self.mm.tree.ctypes_data_access(access.READ),
            self.mm.widths_x.ctypes_data,
            self.mm.widths_y.ctypes_data,
            self.mm.widths_z.ctypes_data,
            self.mm.ncells_x.ctypes_data,
            self.mm.ncells_y.ctypes_data,
            self.mm.ncells_z.ctypes_data,
            ctypes.byref(ie),
        )


        de = REAL(0)

        self._direct_lib(
            INT64(ix),
            INT64(self.mm.max_occ), 
            INT64(self.mm.il_earray.shape[0]),
            self.mm.il_earray.ctypes.get_as_parameter(),
            self.mm.sh.get_pointer(self.positions(access.READ)),
            self.mm.sh.get_pointer(self.charges(access.READ)),
            self.mm.cell_remaps.ctypes.get_as_parameter(),
            self.mm.minc.ctypes.get_as_parameter(),
            self.mm.widths.ctypes.get_as_parameter(),
            self.mm.cell_occ.ctypes.get_as_parameter(),
            self.mm.cell_list.ctypes.get_as_parameter(),
            ctypes.byref(de)
        )

        return ie.value + de.value



    def _init_libs(self):



        g = self.group
        extent = self.domain.extent
        cell_widths = [1.0 / (ex / (sx**(self.R - 1))) for ex, sx in zip(extent, self.mm.subdivision)]


        sph_gen = self.mm.sph_gen

        def cube_ind(L, M):
            return ((L) * ( (L) + 1 ) + (M) )


        assign_gen = ''
        for lx in range(self.L):
            for mx in range(-lx, lx+1):
                reL = SphSymbol('TREE[OFFSET + {ind}]'.format(ind=cube_ind(lx, mx)))
                imL = SphSymbol('TREE[OFFSET + IM_OFFSET + {ind}]'.format(ind=cube_ind(lx, mx)))
                reY, imY = sph_gen.get_y_sym(lx, mx)
                phi_sym = cmplx_mul(reL, imL, reY, imY)[0]
                assign_gen += 'tmp_energy += rhol * ({phi_sym});\n'.format(phi_sym=str(phi_sym))

            assign_gen += 'rhol *= iradius;\n'



        src = r'''
        
        #define REAL double
        #define INT64 int64_t
        #define R                    {R}
        #define EX                   {EX}
        #define EY                   {EY}
        #define EZ                   {EZ}
        #define HEX                  {HEX}
        #define HEY                  {HEY}
        #define HEZ                  {HEZ}
        #define CWX                  {CWX}
        #define CWY                  {CWY}
        #define CWZ                  {CWZ}
        #define LCX                  {LCX}
        #define LCY                  {LCY}
        #define LCZ                  {LCZ}
        #define SDX                  {SDX}
        #define SDY                  {SDY}
        #define SDZ                  {SDZ}
        #define IL_NO                {IL_NO}
        #define IL_STRIDE_OUTER      {IL_STRIDE_OUTER}
        #define NCOMP                {NCOMP}
        #define IM_OFFSET            {IM_OFFSET}
        #define THREE_R              {THREE_R}


        static inline void kernel(
            const REAL  * RESTRICT P,
            const REAL  * RESTRICT Q,
            const INT64 * RESTRICT IL,
            const INT64 * RESTRICT MM_CELLS,
            const INT64 * RESTRICT MM_CHILD_INDEX,
            const REAL  * RESTRICT TREE,
            const REAL  * RESTRICT WIDTHS_X,
            const REAL  * RESTRICT WIDTHS_Y,
            const REAL  * RESTRICT WIDTHS_Z,                            
            const INT64 * RESTRICT NCELLS_X,
            const INT64 * RESTRICT NCELLS_Y,
            const INT64 * RESTRICT NCELLS_Z,                     
                  REAL  * RESTRICT OUT_ENERGY
        ){{

            const double rx = P[0];
            const double ry = P[1];
            const double rz = P[2];

            double particle_energy = 0.0;

            int64_t LEVEL_OFFSETS[R];
            LEVEL_OFFSETS[0] = 0;
            int64_t nx = 1;
            int64_t ny = 1;
            int64_t nz = 1;
            for(int level=1 ; level<R ; level++ ){{
                int64_t nprev = nx * ny * nz * NCOMP;
                LEVEL_OFFSETS[level] = LEVEL_OFFSETS[level - 1] + nprev;
                nx *= SDX;
                ny *= SDY;
                nz *= SDZ;
            }}


            for( int level=0 ; level<R ; level++ ){{

                // cell on this level
                const int64_t cfx = MM_CELLS[level*3 + 0];
                const int64_t cfy = MM_CELLS[level*3 + 1];
                const int64_t cfz = MM_CELLS[level*3 + 2];

                // number of cells on this level
                const int64_t ncx = NCELLS_X[level];
                const int64_t ncy = NCELLS_Y[level];
                const int64_t ncz = NCELLS_Z[level];


                // child on this level
                const int64_t cix = MM_CHILD_INDEX[level * 3 + 0];
                const int64_t ciy = MM_CHILD_INDEX[level * 3 + 1];
                const int64_t ciz = MM_CHILD_INDEX[level * 3 + 2];
                const int64_t ci = cix + SDX * (ciy + SDY * ciz);

                const double wx = WIDTHS_X[level];
                const double wy = WIDTHS_Y[level];
                const double wz = WIDTHS_Z[level];

                
                // loop over IL for this child cell
                for( int ox=0 ; ox<IL_NO ; ox++){{
                    
                    
                    const int64_t ocx = cfx + IL[ci * IL_STRIDE_OUTER + ox * 3 + 0];
                    const int64_t ocy = cfy + IL[ci * IL_STRIDE_OUTER + ox * 3 + 1];
                    const int64_t ocz = cfz + IL[ci * IL_STRIDE_OUTER + ox * 3 + 2];

                    // free space for now
                    if (ocx < 0) {{continue;}}
                    if (ocy < 0) {{continue;}}
                    if (ocz < 0) {{continue;}}
                    if (ocx >= ncx) {{continue;}}
                    if (ocy >= ncy) {{continue;}}
                    if (ocz >= ncz) {{continue;}}

                    const int64_t lin_ind = ocx + NCELLS_X[level] * (ocy + NCELLS_Y[level] * ocz);

                    const double dx = rx - ((-HEX) + (0.5 * wx) + (ocx * wx));
                    const double dy = ry - ((-HEY) + (0.5 * wy) + (ocy * wy));
                    const double dz = rz - ((-HEZ) + (0.5 * wz) + (ocz * wz));

                    const double xy2 = dx * dx + dy * dy;
                    const double radius = sqrt(xy2 + dz * dz);
                    const double theta = atan2(sqrt(xy2), dz);
                    const double phi = atan2(dy, dx);
                    
                    const int64_t OFFSET = LEVEL_OFFSETS[level] + NCOMP * lin_ind;

                    {SPH_GEN}
                    const double iradius = 1.0 / radius;
                    double rhol = iradius;
                    double tmp_energy = 0.0;
                    {ASSIGN_GEN}

                    particle_energy += tmp_energy;

                }}

            }}


            OUT_ENERGY[0] = particle_energy * Q[0];

        }}



        extern "C" int get_energy(
            const INT64            IX,
            const REAL  * RESTRICT P,
            const REAL  * RESTRICT Q,
            const INT64 * RESTRICT IL,
            const INT64 * RESTRICT MM_CELLS,
            const INT64 * RESTRICT MM_CHILD_INDEX,
            const REAL  * RESTRICT TREE,
            const REAL  * RESTRICT WIDTHS_X,
            const REAL  * RESTRICT WIDTHS_Y,
            const REAL  * RESTRICT WIDTHS_Z,                            
            const INT64 * RESTRICT NCELLS_X,
            const INT64 * RESTRICT NCELLS_Y,
            const INT64 * RESTRICT NCELLS_Z,                     
                  REAL  * RESTRICT OUT_ENERGY       
        ){{
            
            
            kernel(
                &P[IX*3],
                &Q[IX],
                IL,
                &MM_CELLS[IX*THREE_R],
                &MM_CHILD_INDEX[IX*THREE_R],
                TREE,
                WIDTHS_X,
                WIDTHS_Y,
                WIDTHS_Z,                            
                NCELLS_X,
                NCELLS_Y,
                NCELLS_Z,                     
                OUT_ENERGY
            );


                
            return 0;
        }}

        '''.format(
            SPH_GEN=str(sph_gen.module),
            ASSIGN_GEN=str(assign_gen),
            R=self.R,
            EX=extent[0],
            EY=extent[1],
            EZ=extent[2],
            HEX=0.5 * extent[0],
            HEY=0.5 * extent[1],
            HEZ=0.5 * extent[2],                
            CWX=cell_widths[0],
            CWY=cell_widths[1],
            CWZ=cell_widths[2],
            LCX=self.mm.subdivision[0] ** (self.R - 1),
            LCY=self.mm.subdivision[1] ** (self.R - 1),
            LCZ=self.mm.subdivision[2] ** (self.R - 1),
            SDX=self.mm.subdivision[0],
            SDY=self.mm.subdivision[1],
            SDZ=self.mm.subdivision[2],
            IL_NO=self.mm.il_array.shape[1],
            IL_STRIDE_OUTER=self.mm.il_array.shape[1] * self.mm.il_array.shape[2],
            NCOMP=self.ncomp,
            IM_OFFSET=self.L**2,
            THREE_R=self.R*3
        )

        self._old_indirect_lib = lib.build.simple_lib_creator('#include <math.h>', src)['get_energy']



        source = r'''
        extern "C" int direct_interactions(
            const INT64            ix,
            const INT64            MAX_OCC, 
            const INT64            NOFFSETS,                //number of nearest neighbour cells
            const INT64 * RESTRICT NNMAP,                   //nearest neighbour offsets
            const REAL  * RESTRICT positions,
            const REAL  * RESTRICT charges,
            const INT64 * RESTRICT cells,
            const INT64 * RESTRICT cell_mins,
            const INT64 * RESTRICT cell_counts,
                  INT64 * RESTRICT cell_occ,
                  INT64 * RESTRICT cell_list,
                  REAL  * RESTRICT total_energy
        ){{ 


            // direct interactions for local charges

            const REAL pix = positions[ix * 3 + 0];
            const REAL piy = positions[ix * 3 + 1];
            const REAL piz = positions[ix * 3 + 2];
            const INT64 cix = cells[ix * 3 + 0];
            const INT64 ciy = cells[ix * 3 + 1];
            const INT64 ciz = cells[ix * 3 + 2];
            const REAL qi = charges[ix];
            
            REAL tmp_energy = 0.0;
            for(INT64 ox=0 ; ox<NOFFSETS ; ox++){{

                INT64 ocx = cix + NNMAP[ox * 3 + 0];
                INT64 ocy = ciy + NNMAP[ox * 3 + 1];
                INT64 ocz = ciz + NNMAP[ox * 3 + 2];
                
                // free space BCs
                if (ocx < 0) {{continue;}}
                if (ocy < 0) {{continue;}}
                if (ocz < 0) {{continue;}}
                if (ocx >= LCX) {{continue;}}
                if (ocy >= LCY) {{continue;}}
                if (ocz >= LCZ) {{continue;}}
                
                ocx -= cell_mins[0];
                ocy -= cell_mins[1];
                ocz -= cell_mins[2];

                // if a plane of edge cells is empty they may not exist in the data structure
                if (ocx < 0) {{continue;}}
                if (ocy < 0) {{continue;}}
                if (ocz < 0) {{continue;}}
                if (ocx >= cell_counts[0]) {{continue;}}
                if (ocy >= cell_counts[1]) {{continue;}}
                if (ocz >= cell_counts[2]) {{continue;}}                   

                const INT64 cj = ocx + cell_counts[0] * (ocy + cell_counts[1] * ocz);

                const int mask = (NNMAP[ox * 3 + 0] == 0) && (NNMAP[ox * 3 + 1] == 0) && (NNMAP[ox * 3 + 2] == 0);
                
                if (mask) {{
                    for(INT64 jxi=0 ; jxi<cell_occ[cj] ; jxi++){{
                        
                        const INT64 jx = cell_list[cj * MAX_OCC + jxi];

                        const REAL pjx = positions[jx * 3 + 0];
                        const REAL pjy = positions[jx * 3 + 1];
                        const REAL pjz = positions[jx * 3 + 2];
                        const REAL qj = charges[jx];


                        const REAL dx = pix - pjx;
                        const REAL dy = piy - pjy;
                        const REAL dz = piz - pjz;

                        const REAL r2 = dx*dx + dy*dy + dz*dz;
                        
                        tmp_energy += (ix == jx) ? 0.0 :  qj / sqrt(r2);
                    

                    }}

                }} else {{
                    for(INT64 jxi=0 ; jxi<cell_occ[cj] ; jxi++){{
                        
                        const INT64 jx = cell_list[cj * MAX_OCC + jxi];

                        const REAL pjx = positions[jx * 3 + 0];
                        const REAL pjy = positions[jx * 3 + 1];
                        const REAL pjz = positions[jx * 3 + 2];
                        const REAL qj = charges[jx];


                        const REAL dx = pix - pjx;
                        const REAL dy = piy - pjy;
                        const REAL dz = piz - pjz;

                        const REAL r2 = dx*dx + dy*dy + dz*dz;
                        
                        tmp_energy += qj / sqrt(r2);

                    }}
                }}

            }}


            total_energy[0] = tmp_energy * qi;

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
            LCX=self.mm.subdivision[0] ** (self.R - 1),
            LCY=self.mm.subdivision[1] ** (self.R - 1),
            LCZ=self.mm.subdivision[2] ** (self.R - 1)
        )

        self._direct_lib = lib.build.simple_lib_creator(header, source)['direct_interactions']



























