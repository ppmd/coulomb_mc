

import numpy as np
from ppmd import data, loop, kernel, access, lib, opt
from ppmd.coulomb import fmm_interaction_lists, octal
from coulomb_kmc import kmc_expansion_tools, common
import ctypes
import math

from coulomb_mc import mc_expansion_tools
import time

Constant = kernel.Constant
REAL = ctypes.c_double
INT64 = ctypes.c_int64

PROFILE = opt.PROFILE

class MCFMM:

    def __init__(self, positions, charges, domain, boundary_condition, r, l):

        self.positions = positions
        self.charges = charges
        self.domain = domain
        self.comm = self.domain.comm
        self.boundary_condition = boundary_condition
        self.R = r
        self.L = l
        self.ncomp = (self.L ** 2) * 2

        self.subdivision = (2, 2, 2)
        
        # tree
        self.tree = octal.OctalTree(self.R, domain.comm)
        self.tree_local = octal.OctalDataTree(self.tree, self.ncomp, 'plain', REAL)
        self.tree_local_ptrs = np.zeros(self.R, ctypes.c_void_p)
        for rx in range(self.R):
            self.tree_local_ptrs[rx] = self.tree_local[rx].ctypes.get_as_parameter().value
        
        # interaction lists
        self.il = fmm_interaction_lists.compute_interaction_lists(domain.extent, self.subdivision)
        self.il_max_len = max(len(lx) for lx in self.il[0])
        
        self.il_array = np.array(self.il[0], INT64)
        self.il_scalararray = data.ScalarArray(ncomp=self.il_array.size, dtype=INT64)
        self.il_scalararray[:] = self.il_array.ravel().copy()

        self.il_earray = np.array(self.il[1], INT64)
        
        # expansion tools
        self.lee = kmc_expansion_tools.LocalExpEval(self.L)
        self.mc_lee = mc_expansion_tools.LocalExp(self.L)


        self.group = self.positions.group
        
        
        pd = type(self.charges)
        g = self.group
        l = self.il_max_len * self.R
        
        # assume that ptr size and int64_t size may be different....

        if ctypes.sizeof(ctypes.c_void_p) == ctypes.sizeof(INT64):
            self.il_pd_ptr_stride = l
            ptr_l = l
        else:
            ptr_l = int(math.ceil((l * ctypes.sizeof(ctypes.c_void_p)) / ctypes.sizeof(INT64)))
            self.il_pd_ptr_stride = int(ptr_l * ctypes.sizeof(INT64) // ctypes.sizeof(ctypes._c_void_p))
        

        g._mc_fmm_cells = pd(ncomp=3, dtype=INT64)
        g._mc_ids = pd(ncomp=1, dtype=INT64)
        g._mc_nexp = pd(ncomp=1, dtype=INT64)
        g._mc_charge = pd(ncomp=l, dtype=REAL)
        g._mc_radius = pd(ncomp=l, dtype=REAL)
        g._mc_theta = pd(ncomp=l, dtype=REAL)
        g._mc_phi = pd(ncomp=l, dtype=REAL)
        g._mc_du = pd(ncomp=l, dtype=REAL)
        g._mc_level = pd(ncomp=l, dtype=INT64)
        g._mc_cx = pd(ncomp=l, dtype=INT64)
        g._mc_cy = pd(ncomp=l, dtype=INT64)
        g._mc_cz = pd(ncomp=l, dtype=INT64)
        g._mc_cl = pd(ncomp=l, dtype=INT64)
        g._mc_ptrs = pd(ncomp=ptr_l, dtype=INT64)
        
        s = self.subdivision
        s = [sx ** (self.R - 1) for sx in s]
        ncells_finest = s[0] * s[1] * s[2]
        self.cell_occupation_ga = data.GlobalArray(ncomp=ncells_finest, dtype=INT64)
        self.cell_occupation = None
        self.cell_indices = np.zeros((s[2], s[1], s[0], 10), INT64)
        self.max_occupancy = None

        self.energy = None

        self.direct_map = {}

        self._cell_bin_loop = None
        self._cell_bin_format_lib = None
        self._init_bin_loop()

        self._direct_contrib_lib = None
        self._init_direct_contrib_lib()
        self._init_indirect_accept_lib()
        self._init_indirect_propose_lib()
    
    def free(self):
        self.tree.free()
        del self.tree_local

    def _init_indirect_propose_lib(self):

        source = r'''

        {INLINE_LOCAL_EXP}

        extern "C" int indirect_propose(
            const REAL  * RESTRICT position,
            const INT64            cellx,
            const INT64            celly,
            const INT64            cellz,
            const REAL             charge,
            const REAL * RESTRICT * RESTRICT moments,
                  REAL * RESTRICT  out
        ){{
            
            // work upwards through the levels

            const REAL px = position[0];
            const REAL py = position[1];
            const REAL pz = position[2];

            INT64 cxo = cellx;
            INT64 cyo = celly;
            INT64 czo = cellz;
            
            REAL tmp_energy = 0.0;

            INT64 CELL_X[R];
            INT64 CELL_Y[R];
            INT64 CELL_Z[R];
            
            for( int rx=R-1 ; rx>=0 ; rx--){{

                CELL_X[rx] = cxo;
                CELL_Y[rx] = cyo;
                CELL_Z[rx] = czo;
                    
                cxo /= SUB_X;
                cyo /= SUB_Y;
                czo /= SUB_Z;
            }}

            
            #pragma omp parallel for schedule(static, 1) reduction(+:tmp_energy)
            for( int rx=R-1 ; rx>=0 ; rx--){{
                const INT64 cx = CELL_X[rx];
                const INT64 cy = CELL_Y[rx];
                const INT64 cz = CELL_Z[rx];

                    
                REAL inner_energy = 0.0;
                const REAL oocx = -HEX + (0.5 + cx) * CELL_WIDTH_X[rx];
                const REAL oocy = -HEY + (0.5 + cy) * CELL_WIDTH_Y[rx];
                const REAL oocz = -HEZ + (0.5 + cz) * CELL_WIDTH_Z[rx];

                const REAL dx = px - oocx;
                const REAL dy = py - oocy;
                const REAL dz = pz - oocz;
                const REAL dx2 = dx*dx;
                const REAL dx2_p_dy2 = dx2 + dy*dy;
                const REAL d2 = dx2_p_dy2 + dz*dz;
                const REAL radius = sqrt(d2);
                const REAL theta = atan2(sqrt(dx2_p_dy2), dz);
                const REAL phi = atan2(dy, dx);

                local_eval(radius, theta, phi, &moments[rx][NCOMP * ( cx + NUM_CELLS_X[rx] * (cy + NUM_CELLS_Y[rx] * cz))], &inner_energy);

                tmp_energy += inner_energy;

                // compute the cell on the next level

            }}
            

            out[0] = tmp_energy * charge;
            return 0;
        }}

        '''.format(
            INLINE_LOCAL_EXP=self.mc_lee.create_single_local_eval_src
        )


        e = self.domain.extent
        s = self.subdivision
        R = self.R

        header = r'''
        #include <math.h>
        #include <stdio.h>
        #define REAL double
        #define INT64 int64_t
        #define R {R}
        #define HEX {HEX}
        #define HEY {HEY}
        #define HEZ {HEZ}
        #define NCOMP {NCOMP}
        
        #define SUB_X {SUB_X}
        #define SUB_Y {SUB_Y}
        #define SUB_Z {SUB_Z}

        const REAL CELL_WIDTH_X[R] = {{ {CELL_WIDTH_X} }};
        const REAL CELL_WIDTH_Y[R] = {{ {CELL_WIDTH_Y} }};
        const REAL CELL_WIDTH_Z[R] = {{ {CELL_WIDTH_Z} }};        
        const INT64 NUM_CELLS_X[R] = {{ {NUM_CELLS_X} }};
        const INT64 NUM_CELLS_Y[R] = {{ {NUM_CELLS_Y} }};
        const INT64 NUM_CELLS_Z[R] = {{ {NUM_CELLS_Z} }};

        '''.format(
            R=self.R,
            HEX=e[0] * 0.5,
            HEY=e[1] * 0.5,
            HEZ=e[2] * 0.5,
            SUB_X=s[0],
            SUB_Y=s[1],
            SUB_Z=s[2],
            CELL_WIDTH_X=','.join([str(e[0] / (s[0] ** (rx))) for rx in range(R)]),
            CELL_WIDTH_Y=','.join([str(e[1] / (s[1] ** (rx))) for rx in range(R)]),
            CELL_WIDTH_Z=','.join([str(e[2] / (s[2] ** (rx))) for rx in range(R)]),
            NUM_CELLS_X=','.join([str(int(s[0] ** (rx))) for rx in range(R)]),
            NUM_CELLS_Y=','.join([str(int(s[1] ** (rx))) for rx in range(R)]),
            NUM_CELLS_Z=','.join([str(int(s[2] ** (rx))) for rx in range(R)]),
            NCOMP=self.ncomp
        )

        self._indirect_propose_lib = lib.build.simple_lib_creator(header, source)['indirect_propose']



    def _init_indirect_accept_lib(self):

        source = r'''

        {INLINE_LOCAL_EXP}

        extern "C" int indirect_accept(
            const REAL  * RESTRICT old_position,
            const REAL  * RESTRICT new_position,
            const REAL             charge,
            const INT64 * RESTRICT offsets,
            REAL * RESTRICT * RESTRICT moments
        ){{
            


            #pragma omp parallel for collapse(2)
            for( int rx=0 ; rx<R ; rx++){{
                for( int ox=0 ; ox<NUM_OFFSETS ; ox++){{
                    

                    const REAL ospx = old_position[0] + HEX;
                    const REAL ospy = old_position[1] + HEY;
                    const REAL ospz = old_position[2] + HEZ;
                    
                    // get the cell
                    const int ocx = ospx * CELL_BIN_X[rx];
                    const int ocy = ospy * CELL_BIN_Y[rx];
                    const int ocz = ospz * CELL_BIN_Z[rx];
                    
                    // compute the child index
                    const int ccx = ocx % SUB_X;
                    const int ccy = ocy % SUB_Y;
                    const int ccz = ocz % SUB_Z;
                    const int child_index = ccx + SUB_X * (ccy + SUB_Y * ccz);
                    
                    const int oscx = ocx + offsets[child_index * 3 * NUM_OFFSETS + ox * 3 + 0];
                    const int oscy = ocy + offsets[child_index * 3 * NUM_OFFSETS + ox * 3 + 1];
                    const int oscz = ocz + offsets[child_index * 3 * NUM_OFFSETS + ox * 3 + 2];
                    
                    // free space checking

                    if (
                        (oscx < 0)  ||
                        (oscy < 0)  ||
                        (oscz < 0)  ||
                        (oscx >= NUM_CELLS_X[rx]) ||
                        (oscy >= NUM_CELLS_Y[rx]) ||
                        (oscz >= NUM_CELLS_Z[rx])
                    ) {{
                        continue;
                    }}
                    

                    const REAL oocx = -HEX + (0.5 + oscx) * CELL_WIDTH_X[rx];
                    const REAL oocy = -HEY + (0.5 + oscy) * CELL_WIDTH_Y[rx];
                    const REAL oocz = -HEZ + (0.5 + oscz) * CELL_WIDTH_Z[rx];

                    REAL dx = old_position[0] - oocx;
                    REAL dy = old_position[1] - oocy;
                    REAL dz = old_position[2] - oocz;
                    REAL dx2 = dx*dx;
                    REAL dx2_p_dy2 = dx2 + dy*dy;
                    REAL d2 = dx2_p_dy2 + dz*dz;
                    REAL radius = sqrt(d2);
                    REAL theta = atan2(sqrt(dx2_p_dy2), dz);
                    REAL phi = atan2(dy, dx);

                    inline_local_exp(-1.0 * charge, radius, theta, phi, &moments[rx][NCOMP * ( oscx + NUM_CELLS_X[rx] * (oscy + NUM_CELLS_Y[rx] * oscz))]);

                }}
            }}


            #pragma omp parallel for collapse(2)
            for( int rx=0 ; rx<R ; rx++){{
                for( int ox=0 ; ox<NUM_OFFSETS ; ox++){{
                    

                    const REAL ospx = new_position[0] + HEX;
                    const REAL ospy = new_position[1] + HEY;
                    const REAL ospz = new_position[2] + HEZ;
                    
                    const int ocx = ospx * CELL_BIN_X[rx];
                    const int ocy = ospy * CELL_BIN_Y[rx];
                    const int ocz = ospz * CELL_BIN_Z[rx];
                    
                    // compute the child index
                    const int ccx = ocx % SUB_X;
                    const int ccy = ocy % SUB_Y;
                    const int ccz = ocz % SUB_Z;
                    const int child_index = ccx + SUB_X * (ccy + SUB_Y * ccz);
                    
                    const int oscx = ocx + offsets[child_index * 3 * NUM_OFFSETS + ox * 3 + 0];
                    const int oscy = ocy + offsets[child_index * 3 * NUM_OFFSETS + ox * 3 + 1];
                    const int oscz = ocz + offsets[child_index * 3 * NUM_OFFSETS + ox * 3 + 2];
                    
                    // free space checking

                    if (
                        (oscx < 0)  ||
                        (oscy < 0)  ||
                        (oscz < 0)  ||
                        (oscx >= NUM_CELLS_X[rx]) ||
                        (oscy >= NUM_CELLS_Y[rx]) ||
                        (oscz >= NUM_CELLS_Z[rx])
                    ) {{
                        continue;
                    }}


                    const REAL oocx = -HEX + (0.5 + oscx) * CELL_WIDTH_X[rx];
                    const REAL oocy = -HEY + (0.5 + oscy) * CELL_WIDTH_Y[rx];
                    const REAL oocz = -HEZ + (0.5 + oscz) * CELL_WIDTH_Z[rx];

                    REAL dx = new_position[0] - oocx;
                    REAL dy = new_position[1] - oocy;
                    REAL dz = new_position[2] - oocz;
                    REAL dx2 = dx*dx;
                    REAL dx2_p_dy2 = dx2 + dy*dy;
                    REAL d2 = dx2_p_dy2 + dz*dz;
                    REAL radius = sqrt(d2);
                    REAL theta = atan2(sqrt(dx2_p_dy2), dz);
                    REAL phi = atan2(dy, dx);

                    inline_local_exp(charge, radius, theta, phi, &moments[rx][NCOMP * ( oscx + NUM_CELLS_X[rx] * (oscy + NUM_CELLS_Y[rx] * oscz))]);

                }}
            }}






            return 0;
        }}

        '''.format(
            INLINE_LOCAL_EXP=self.mc_lee.create_local_exp_src
        )
        


        e = self.domain.extent
        s = self.subdivision
        R = self.R

        header = r'''
        #include <math.h>
        #include <stdio.h>
        #define REAL double
        #define INT64 int64_t
        #define R {R}
        #define NUM_OFFSETS {NUM_OFFSETS}
        #define HEX {HEX}
        #define HEY {HEY}
        #define HEZ {HEZ}
        #define NCOMP {NCOMP}
        
        #define SUB_X {SUB_X}
        #define SUB_Y {SUB_Y}
        #define SUB_Z {SUB_Z}

        const REAL CELL_BIN_X[R] = {{ {CELL_BIN_X} }};
        const REAL CELL_BIN_Y[R] = {{ {CELL_BIN_Y} }};
        const REAL CELL_BIN_Z[R] = {{ {CELL_BIN_Z} }};
        const REAL CELL_WIDTH_X[R] = {{ {CELL_WIDTH_X} }};
        const REAL CELL_WIDTH_Y[R] = {{ {CELL_WIDTH_Y} }};
        const REAL CELL_WIDTH_Z[R] = {{ {CELL_WIDTH_Z} }};        
        const INT64 NUM_CELLS_X[R] = {{ {NUM_CELLS_X} }};
        const INT64 NUM_CELLS_Y[R] = {{ {NUM_CELLS_Y} }};
        const INT64 NUM_CELLS_Z[R] = {{ {NUM_CELLS_Z} }};

        '''.format(
            R=self.R,
            NUM_OFFSETS=self.il_max_len,
            HEX=e[0] * 0.5,
            HEY=e[1] * 0.5,
            HEZ=e[2] * 0.5,
            SUB_X=s[0],
            SUB_Y=s[1],
            SUB_Z=s[2],
            CELL_BIN_X=','.join([str((s[0] ** (rx)) / e[0]) for rx in range(R)]),
            CELL_BIN_Y=','.join([str((s[1] ** (rx)) / e[1]) for rx in range(R)]),
            CELL_BIN_Z=','.join([str((s[2] ** (rx)) / e[2]) for rx in range(R)]),            
            CELL_WIDTH_X=','.join([str(e[0] / (s[0] ** (rx))) for rx in range(R)]),
            CELL_WIDTH_Y=','.join([str(e[1] / (s[1] ** (rx))) for rx in range(R)]),
            CELL_WIDTH_Z=','.join([str(e[2] / (s[2] ** (rx))) for rx in range(R)]),
            NUM_CELLS_X=','.join([str(int(s[0] ** (rx))) for rx in range(R)]),
            NUM_CELLS_Y=','.join([str(int(s[1] ** (rx))) for rx in range(R)]),
            NUM_CELLS_Z=','.join([str(int(s[2] ** (rx))) for rx in range(R)]),
            NCOMP=self.ncomp
        )

        self._indirect_accept_lib = lib.build.simple_lib_creator(header, source)['indirect_accept']



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


    def _init_bin_loop(self):
        
        g = self.group
        extent = self.domain.extent
        cell_widths = [1.0 / (ex / (sx**(self.R - 1))) for ex, sx in zip(extent, self.subdivision)]

        k = kernel.Kernel(
            'mc_bin_loop',
            r'''
            
            const double rx = P.i[0];
            const double ry = P.i[1];
            const double rz = P.i[2];

            // bin into finest level cell
            const double srx = rx + HEX;
            const double sry = ry + HEY;
            const double srz = rz + HEZ;
            
            int64_t cfx = srx * CWX;
            int64_t cfy = sry * CWY;
            int64_t cfz = srz * CWZ;

            cfx = (cfx < LCX) ? cfx : (LCX - 1);
            cfy = (cfy < LCX) ? cfy : (LCY - 1);
            cfz = (cfz < LCX) ? cfz : (LCZ - 1);
            
            // record the finest level cells
            MC_FC.i[0] = cfx;
            MC_FC.i[1] = cfy;
            MC_FC.i[2] = cfz;
            
            // number of cells in each direction
            int64_t ncx = LCX;
            int64_t ncy = LCY;
            int64_t ncz = LCZ;
            
            // increment the occupancy for this cell
            OCC_GA[cfx + LCX * (cfy + LCY * cfz)]++;

            int64_t n = 0;
            for( int level=R-1 ; level>=0 ; level-- ){{
                
                // cell widths for cell centre computation
                const double wx = EX / ncx;
                const double wy = EY / ncy;
                const double wz = EZ / ncz;

                // child on this level

                const int64_t cix = cfx % SDX;
                const int64_t ciy = cfy % SDY;
                const int64_t ciz = cfz % SDZ;


                const int64_t ci = cix + SDX * (ciy + SDY * ciz);
                
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

                    MC_CX.i[n] = ocx;
                    MC_CY.i[n] = ocy;
                    MC_CZ.i[n] = ocz;
                    MC_CL.i[n] = ocx + ncx * (ocy + ncy * ocz);
                    MC_LEVEL.i[n] = level;

                    MC_DX.i[n] = rx - ((-HEX) + (0.5 * wx) + (ocx * wx));
                    MC_DY.i[n] = ry - ((-HEY) + (0.5 * wy) + (ocy * wy));
                    MC_DZ.i[n] = rz - ((-HEZ) + (0.5 * wz) + (ocz * wz));

                    n++;
                }}

                // compute the cells for the next level
                cfx /= SDX;
                cfy /= SDY;
                cfz /= SDZ;

                // number of cells in each dim for the next level
                ncx /= SDX;
                ncy /= SDY;
                ncz /= SDZ;

            }}

            MC_NEXP.i[0] = n;

            // compute offsets as spherical coordinates in a vcectorisable loop
            for( int ox=0 ; ox<n ; ox++){{
                const double dx = MC_DX.i[ox];
                const double dy = MC_DY.i[ox];
                const double dz = MC_DZ.i[ox];

                const double xy2 = dx * dx + dy * dy;
                MC_DX.i[ox] = sqrt(xy2 + dz * dz);
                MC_DY.i[ox] = atan2(sqrt(xy2), dz);
                MC_DZ.i[ox] = atan2(dy, dx);
                MC_CHR.i[ox] = Q.i[0];
            }}

            '''.format(
            ),
            (
                Constant('R', self.R),
                Constant('EX', extent[0]),
                Constant('EY', extent[1]),
                Constant('EZ', extent[2]),
                Constant('HEX', 0.5 * extent[0]),
                Constant('HEY', 0.5 * extent[1]),
                Constant('HEZ', 0.5 * extent[2]),                
                Constant('CWX', cell_widths[0]),
                Constant('CWY', cell_widths[1]),
                Constant('CWZ', cell_widths[2]),
                Constant('LCX', self.subdivision[0] ** (self.R - 1)),
                Constant('LCY', self.subdivision[1] ** (self.R - 1)),
                Constant('LCZ', self.subdivision[2] ** (self.R - 1)),
                Constant('SDX', self.subdivision[0]),
                Constant('SDY', self.subdivision[1]),
                Constant('SDZ', self.subdivision[2]),
                Constant('IL_NO', self.il_array.shape[1]),
                Constant('IL_STRIDE_OUTER', self.il_array.shape[1] * self.il_array.shape[2]),
            )
        )

        dat_dict = {
            'P': self.positions(access.READ),
            'Q': self.charges(access.READ),
            'IL': self.il_scalararray(access.READ),
            'MC_FC': g._mc_fmm_cells(access.WRITE),
            'MC_NEXP': g._mc_nexp(access.WRITE),
            'MC_CHR': g._mc_charge(access.WRITE),
            'MC_DX': g._mc_radius(access.WRITE),
            'MC_DY': g._mc_theta(access.WRITE),
            'MC_DZ': g._mc_phi(access.WRITE),
            'MC_LEVEL': g._mc_level(access.WRITE),
            'MC_CX': g._mc_cx(access.WRITE),
            'MC_CY': g._mc_cy(access.WRITE),
            'MC_CZ': g._mc_cz(access.WRITE),
            'MC_CL': g._mc_cl(access.WRITE),
            'OCC_GA': self.cell_occupation_ga(access.INC_ZERO),
        }

        self._cell_bin_loop = loop.ParticleLoopOMP(kernel=k, dat_dict=dat_dict)

        # create the lib that takes the particledat data and produces the data for the local expansion lib

        source = '''
        extern "C" int bookkeeping_entry(
            const int64_t n,            // number of particles
            const int64_t l,            // normal stride
            const int64_t mc_cl_stride, // pointer dat stride
            const int64_t * MC_NEXP,
            const int64_t * MC_LEVEL,
            const int64_t * MC_CL,      // linear index on each level
            double ** start_ptrs,       // the starting pointer for each level
            double ** MC_PTRS 
        ){{
            int err = 0;
            
            #pragma omp parallel for
            for( int ix=0 ; ix<n ; ix++ ){{
                for(int ox=0 ; ox<MC_NEXP[ix] ; ox++){{
                    MC_PTRS[ix * mc_cl_stride + ox] = start_ptrs[MC_LEVEL[ix * l + ox]] + (MC_CL[ix * l + ox] * {NCOMP});
                }}
            }}

            return err;
        }}
        '''.format(
            NCOMP=self.ncomp
        )

        header = '''
        #include <stdio.h>
        '''.format()

        self._cell_bin_format_lib = lib.build.simple_lib_creator(header, source)['bookkeeping_entry']

    

    



    

    def _get_cell(self, position):

        extent = self.domain.extent
        cell_widths = [ex / (2**(self.R - 1)) for ex in extent]
        spos = [0.5*ex + po for po, ex in zip(position, extent)]
        
        # if a charge is slightly out of the negative end of an axis this will
        # truncate to zero
        cell = [int(pcx / cwx) for pcx, cwx in zip(spos, cell_widths)]
        # truncate down if too high on axis, if way too high this should 
        # probably throw an error
        cc = tuple([min(cx, (2**(self.R -1))-1) for cx in cell ])
        return cc


    def _get_cell_disp(self, cell, position, level):
        """
        Returns spherical coordinate of particle with local cell centre as an
        origin
        """
        extent = self.domain.extent
        sl = 2 ** level
        csl = [extent[0] / sl,
               extent[1] / sl,
               extent[2] / sl]
        
        es = [extent[0] * -0.5,
              extent[1] * -0.5,
              extent[2] * -0.5]

        ec = [esx + 0.5 * cx + ccx * cx for esx, cx, ccx in zip(es, csl, cell)]
        
        disp = (position[0] - ec[0], position[1] - ec[1], position[2] - ec[2])
        sph = common.spherical(disp)
        
        return sph
 

    def _get_parent(self, cell, level):
        """
        Return the index of the parent of a cell on a level
        """
        c = 2**(self.R - level - 1)
        return (
            int(cell[0]) // c,
            int(cell[1]) // c,
            int(cell[2]) // c
        )


    def _get_child_index(self, cell):

        return (int(cell[0]) % 2) + 2 * ((int(cell[1]) % 2) + 2 * (int(cell[2]) % 2))


    def _setup_tree(self):
        
        for levelx in range(self.R):
            self.tree_local[levelx][:] = 0

        g = self.positions.group
        N = self.positions.npart_local

        start_ptrs = np.zeros(self.R, ctypes.c_void_p)
        for rx in range(self.R):
            start_ptrs[rx] = self.tree_local[rx].ctypes.get_as_parameter().value

        self._cell_bin_loop.execute()
        self._profile_inc('_cell_bin_wrapper', self._cell_bin_loop.wrapper_timer.time())

        self.max_occupancy = np.max(self.cell_occupation_ga[:])
        s = self.subdivision
        s = [sx ** (self.R - 1) for sx in s]
        self.cell_occupation = self.cell_occupation_ga[:].copy().reshape((s[2], s[1], s[0]))

        mc_nexp = g._mc_nexp.view
        mc_charge = g._mc_charge.view
        mc_radius = g._mc_radius.view
        mc_theta = g._mc_theta.view
        mc_phi = g._mc_phi.view
        mc_ptrs = g._mc_ptrs.view
        mc_level = g._mc_level.view
        mc_cl = g._mc_cl.view

        self._cell_bin_format_lib(
            INT64(N),
            INT64(self.il_max_len * self.R),
            INT64(self.il_pd_ptr_stride),
            mc_nexp.ctypes.get_as_parameter(),
            mc_level.ctypes.get_as_parameter(),
            mc_cl.ctypes.get_as_parameter(),
            start_ptrs.ctypes.get_as_parameter(),
            mc_ptrs.ctypes.get_as_parameter()
        )


        for px in range(N):
            self.mc_lee.compute_local_exp(
                mc_nexp[px, 0],
                mc_charge[px,:],
                mc_radius[px, :],
                mc_theta[px, :],
                mc_phi[px, :],
                mc_ptrs[px, :]
            )

        return
        # to pass into lib
        charge = np.zeros(self.il_max_len * self.R, REAL)
        radius = np.zeros_like(charge)
        theta = np.zeros_like(charge)
        phi = np.zeros_like(charge)
        ptrs = np.zeros(self.il_max_len * self.R, ctypes.c_void_p)


        with g._mc_fmm_cells.modify_view() as gm:

            for px in range(N):

                n = 0
                pos = self.positions[px,:].copy()
                q = float(self.charges[px, 0])

                # cell on finest level
                cell = self._get_cell(pos)

                # CE+LL WRITING DISABLED
                #gm[px, :] = cell
                if tuple(cell) != tuple(gm[px, :]): import ipdb; ipdb.set_trace()

                for level in range(self.R-1, -1, -1):
                    
                    # cell on level 
                    cell_level = self._get_parent(cell, level)
                    child_index = self._get_child_index(cell_level)

                    il = self.il[0][child_index]

                    for ox in il:
                        ccc = (
                            cell_level[0] + ox[0], 
                            cell_level[1] + ox[1], 
                            cell_level[2] + ox[2], 
                        )

                        # test if outside domain
                        sl = 2**level
                        if ccc[0] < 0: continue
                        if ccc[1] < 0: continue
                        if ccc[2] < 0: continue
                        if ccc[0] >= sl: continue
                        if ccc[1] >= sl: continue
                        if ccc[2] >= sl: continue

                        sph = self._get_cell_disp(ccc, pos, level)
                        #self.lee.local_exp(sph, q, self.tree_local[level][ccc[2], ccc[1], ccc[0], :])

                        charge[n] = q
                        radius[n] = sph[0]
                        theta[n] = sph[1]
                        phi[n] = sph[2]
                        ptrs[n] = self.tree_local[level][ccc[2], ccc[1], ccc[0], :].ctypes.get_as_parameter().value
                        n += 1

                err = np.linalg.norm(charge[:n] - g._mc_charge[px, :].ravel()[:n], np.inf)
                print(err)
                if err > 10.**-12: import ipdb; ipdb.set_trace()
                err = np.linalg.norm(radius[:n] - g._mc_radius[px, :].ravel()[:n], np.inf)
                print(err)
                if err > 10.**-12: import ipdb; ipdb.set_trace()
                err = np.linalg.norm(theta[:n] - g._mc_theta[px, :].ravel()[:n], np.inf)
                print(err)
                err = np.linalg.norm(phi[:n] - g._mc_phi[px, :].ravel()[:n], np.inf)
                print(err)               
                if err > 10.**-12: import ipdb; ipdb.set_trace()
                if g._mc_nexp[px, 0] != n: import ipdb; ipdb.set_trace()

                aaa = g._mc_ptrs[px, :].view(dtype=ctypes.c_void_p)
                err = np.linalg.norm(aaa[:n].ravel() - ptrs[:n], np.inf)
                print(err)               
                if err > 10.**-12: import ipdb; ipdb.set_trace()

                # EXPANSION COMPUTATION DISABLEED
                #self.mc_lee.compute_local_exp(n, charge, radius, theta, phi, ptrs)



    
    def _compute_energy(self):

        N = self.positions.npart_local
        g = self.positions.group


        C = g._mc_fmm_cells

        energy = 0.0
        
        self.direct_map = {}
        # indirect part ( and bin into cells)
        for px in range(N):

            # cell on finest level
            cell = tuple(C[px, :])

            if cell in self.direct_map.keys():
                self.direct_map[cell].append(px)
            else:
                self.direct_map[cell] = [px]

        self._make_occupancy_map()

        
        with g._mc_ids.modify_view() as mv:
            mv[:, 0] = np.arange(N)


        # indirect part 
        energy_py = 0.0
        for px in range(N):
            tmp = self._get_old_energy(px)
            energy_py += tmp

        energy += energy_py

        return 0.5 * energy
    

    def _get_old_energy(self, px):

        tmp = np.zeros(self.ncomp, REAL)
        N = self.positions.npart_local
        g = self.positions.group
        C = g._mc_fmm_cells
        R = self.R

        sl = 2**(self.R - 1)
        energy = 0.0

        pos = self.positions[px,:].copy()

        charge = float(self.charges[px, 0])

        # cell on finest level
        cell = (C[px, 0], C[px, 1], C[px, 2])
        
        #radius = np.zeros(R, REAL)
        #theta = np.zeros(R, REAL)
        #phi = np.zeros(R, REAL)
        #moments = np.zeros(R, ctypes.c_void_p)
        #out = np.zeros(R, REAL)
        #

        #for level in range(self.R):
        #    
        #    cell_level = self._get_parent(cell, level)
        #    sph = self._get_cell_disp(cell_level, pos, level)

        #    radius[level] = sph[0]
        #    theta[level] = sph[1]
        #    phi[level] = sph[2]
        #    moments[level] = self.tree_local[level][cell_level[2], cell_level[1], cell_level[0], :].ctypes.get_as_parameter().value
        #
        #self.mc_lee.compute_phi_local(R, radius, theta, phi, moments, out)
        # energy += np.sum(out) * charge
        
        t0 = time.time()
        energy_c = REAL(0)
        self._indirect_propose_lib(
            pos.ctypes.get_as_parameter(),
            INT64(cell[0]),
            INT64(cell[1]),
            INT64(cell[2]),
            REAL(charge),
            self.tree_local_ptrs.ctypes.get_as_parameter(),
            ctypes.byref(energy_c)
        )
        energy += energy_c.value
        self._profile_inc('indirect_old', time.time() - t0)


        energy_pc = REAL()
        index_c = INT64(px)
        
        t0 = time.time()
        self._direct_contrib_lib(
            INT64(0),                         #// -1 for "all old contribs (does not use id array)", 0 for indexed old contribs, 1 for new contribs
            INT64(1),                          #//number of contributions to compute
            INT64(self.il_earray.shape[0]),
            INT64(self.cell_indices.shape[3]),
            ctypes.byref(index_c),
            g._mc_ids.view.ctypes.get_as_parameter(),
            self.positions.view.ctypes.get_as_parameter(),
            self.charges.view.ctypes.get_as_parameter(),
            g._mc_fmm_cells.view.ctypes.get_as_parameter(),
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

        #energy_py = 0.0
        #for ox in self.il[1]:

        #    ccc = (
        #        cell[0] + ox[0],
        #        cell[1] + ox[1],
        #        cell[2] + ox[2],
        #    )
        #    if ccc[0] < 0: continue
        #    if ccc[1] < 0: continue
        #    if ccc[2] < 0: continue
        #    if ccc[0] >= sl: continue
        #    if ccc[1] >= sl: continue
        #    if ccc[2] >= sl: continue

        #    ccc = tuple(ccc)
        #    if ccc not in self.direct_map: continue

        #    for jx in self.direct_map[ccc]:
        #        if jx == px: continue

        #        energy_py += charge * self.charges[jx, 0] / np.linalg.norm(pos - self.positions[jx, :])
        #energy += energy_py

        return energy


    def _get_new_energy(self, px, pos):

        tmp = np.zeros(self.ncomp, REAL)
        N = self.positions.npart_local
        g = self.positions.group
        R = self.R       
        sl = 2**(self.R - 1)
        energy = 0.0

        charge = float(self.charges[px, 0])

        # cell on finest level
        cell = self._get_cell(pos)

        #radius = np.zeros(R, REAL)
        #theta = np.zeros(R, REAL)
        #phi = np.zeros(R, REAL)
        #moments = np.zeros(R, ctypes.c_void_p)
        #out = np.zeros(R, REAL)

        #for level in range(self.R):
        #    
        #    cell_level = self._get_parent(cell, level)
        #    sph = self._get_cell_disp(cell_level, pos, level)

        #    radius[level] = sph[0]
        #    theta[level] = sph[1]
        #    phi[level] = sph[2]
        #    moments[level] = self.tree_local[level][cell_level[2], cell_level[1], cell_level[0], :].ctypes.get_as_parameter().value
        #
        #self.mc_lee.compute_phi_local(R, radius, theta, phi, moments, out)
        #energy += np.sum(out) * charge

        t0 = time.time()
        energy_c = REAL(0)
        self._indirect_propose_lib(
            pos.ctypes.get_as_parameter(),
            INT64(cell[0]),
            INT64(cell[1]),
            INT64(cell[2]),
            REAL(charge),
            self.tree_local_ptrs.ctypes.get_as_parameter(),
            ctypes.byref(energy_c)
        )
        energy += energy_c.value
        self._profile_inc('indirect_new', time.time() - t0)


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
            g._mc_ids.view.ctypes.get_as_parameter(),
            self.positions.view.ctypes.get_as_parameter(),
            self.charges.view.ctypes.get_as_parameter(),
            g._mc_fmm_cells.view.ctypes.get_as_parameter(),
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

        #energy_py = 0.0
        #for ox in self.il[1]:
        #    ccc = (
        #        cell[0] + ox[0],
        #        cell[1] + ox[1],
        #        cell[2] + ox[2],
        #    )
        #    if ccc[0] < 0: continue
        #    if ccc[1] < 0: continue
        #    if ccc[2] < 0: continue
        #    if ccc[0] >= sl: continue
        #    if ccc[1] >= sl: continue
        #    if ccc[2] >= sl: continue

        #    ccc = tuple(ccc)
        #    if ccc not in self.direct_map: continue

        #    for jx in self.direct_map[ccc]:
        #        energy_py += charge * self.charges[jx, 0] / np.linalg.norm(pos - self.positions[jx, :])


        #energy += energy_py


        return energy
    

    def _get_self_interaction(self, px, pos):

        charge = self.charges[px, 0]
        old_pos = self.positions[px, :]

        return charge * charge / np.linalg.norm(old_pos.ravel() - pos.ravel())


    def initialise(self):
        self._setup_tree()
        self.energy = self._compute_energy()

    
    def propose(self, move):
        px = int(move[0])
        pos = move[1]
        
        old_energy = self._get_old_energy(px)
        new_energy = self._get_new_energy(px, pos)
        self_energy = self._get_self_interaction(px, pos)
        #print("\t-->", old_energy, new_energy, self_energy)

        self._profile_inc('num_propose', 1)
        return  new_energy - old_energy - self_energy


    def _make_occupancy_map(self):
        s = self.subdivision
        s = [sx ** (self.R - 1) for sx in s]
        if self.cell_indices.shape[3] < self.max_occupancy:
            self.cell_indices = np.zeros((s[2], s[1], s[0], self.max_occupancy), INT64)
        
        for cellx in self.direct_map.keys():
            l = len(self.direct_map[cellx])
            self.cell_indices[cellx[2], cellx[1], cellx[0], :l] = self.direct_map[cellx]





    def accept(self, move, energy_diff=None):
        t0 = time.time()
        px = int(move[0])
        new_pos = move[1]


        if energy_diff is None:
            energy_diff = self.propose(move)
        
        self.energy += energy_diff


        old_pos = self.positions[px, :].copy()
        q = float(self.charges[px, 0])
        
        g = self.positions.group
        old_cell = g._mc_fmm_cells[px, :].copy()
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
        
        new_position = np.array((new_pos[0], new_pos[1], new_pos[2]), REAL)
        old_position = np.array((old_pos[0], old_pos[1], old_pos[2]), REAL)

        self._profile_inc('direct_accept', time.time() - t0)       

        t0 = time.time()
        self._indirect_accept_lib(
            old_position.ctypes.get_as_parameter(),
            new_position.ctypes.get_as_parameter(),
            REAL(q),
            self.il_array.ctypes.get_as_parameter(),
            self.tree_local_ptrs.ctypes.get_as_parameter()
        )
        self._profile_inc('indirect_accept', time.time() - t0)

        ### to pass into lib
        #charge = np.zeros(self.il_max_len * self.R, REAL)
        #radius = np.zeros_like(charge)
        #theta = np.zeros_like(charge)
        #phi = np.zeros_like(charge)
        #ptrs = np.zeros(self.il_max_len * self.R, ctypes.c_void_p)

        #n = 0
        ## remove old contrib
        #for level in range(self.R):
        #    
        #    # cell on level 
        #    cell_level = self._get_parent(old_cell, level)
        #    child_index = self._get_child_index(cell_level)

        #    il = self.il[0][child_index]

        #    for ox in il:
        #        ccc = (
        #            cell_level[0] + ox[0], 
        #            cell_level[1] + ox[1], 
        #            cell_level[2] + ox[2], 
        #        )

        #        # test if outside domain
        #        sl = 2**level
        #        if ccc[0] < 0: continue
        #        if ccc[1] < 0: continue
        #        if ccc[2] < 0: continue
        #        if ccc[0] >= sl: continue
        #        if ccc[1] >= sl: continue
        #        if ccc[2] >= sl: continue

        #        sph = self._get_cell_disp(ccc, old_pos, level)
        #        #self.lee.local_exp(sph, -q, self.tree_local[level][ccc[2], ccc[1], ccc[0], :])

        #        charge[n] = -q
        #        radius[n] = sph[0]
        #        theta[n] = sph[1]
        #        phi[n] = sph[2]
        #        ptrs[n] = self.tree_local[level][ccc[2], ccc[1], ccc[0], :].ctypes.get_as_parameter().value
        #        n += 1

        #self.mc_lee.compute_local_exp(n, charge, radius, theta, phi, ptrs)

        #n = 0
        ## add new contrib
        #for level in range(self.R):
        #    
        #    # cell on level 
        #    cell_level = self._get_parent(new_cell, level)
        #    child_index = self._get_child_index(cell_level)

        #    il = self.il[0][child_index]

        #    for ox in il:
        #        ccc = (
        #            cell_level[0] + ox[0], 
        #            cell_level[1] + ox[1], 
        #            cell_level[2] + ox[2], 
        #        )

        #        # test if outside domain
        #        sl = 2**level
        #        if ccc[0] < 0: continue
        #        if ccc[1] < 0: continue
        #        if ccc[2] < 0: continue
        #        if ccc[0] >= sl: continue
        #        if ccc[1] >= sl: continue
        #        if ccc[2] >= sl: continue

        #        sph = self._get_cell_disp(ccc, new_pos, level)
        #        #self.lee.local_exp(sph, q, self.tree_local[level][ccc[2], ccc[1], ccc[0], :])

        #        charge[n] = q
        #        radius[n] = sph[0]
        #        theta[n] = sph[1]
        #        phi[n] = sph[2]
        #        ptrs[n] = self.tree_local[level][ccc[2], ccc[1], ccc[0], :].ctypes.get_as_parameter().value
        #        n += 1

        #self.mc_lee.compute_local_exp(n, charge, radius, theta, phi, ptrs)


        t0 = time.time()
        assert self.comm.size == 1
        with g._mc_fmm_cells.modify_view() as m:
            m[px, :] = new_cell

        
        # move the particle in the dats
        with self.positions.modify_view() as m:
            m[px, :] = new_pos.copy()

        self._profile_inc('dats_accept', time.time() - t0)
        self._profile_inc('num_accept', 1)



    def _profile_inc(self, key, inc):
        key = self.__class__.__name__ + ':' + key
        if key not in PROFILE.keys():
            PROFILE[key] = inc
        else:
            PROFILE[key] += inc

    def _profile_get(self, key):
        key = self.__class__.__name__ + ':' + key
        return PROFILE[key]

    def _profile_set(self, key, inc):
        key = self.__class__.__name__ + ':' + key
        if key not in PROFILE.keys():
            PROFILE[key] = inc
        else:
            PROFILE[key] = inc 




 

