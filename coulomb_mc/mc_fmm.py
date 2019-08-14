

import numpy as np
from ppmd import data, loop, kernel, access, lib, opt
from ppmd.coulomb import fmm_interaction_lists, octal
from coulomb_kmc import kmc_expansion_tools, common
import ctypes
import math

from coulomb_mc import mc_expansion_tools
from coulomb_mc.mc_direct import DirectCommon
from coulomb_mc.mc_common import MCCommon


import time

Constant = kernel.Constant
REAL = ctypes.c_double
INT64 = ctypes.c_int64

PROFILE = opt.PROFILE



class MCFMM(MCCommon):

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
        
        
        self.tree_local_ga = data.GlobalArray(ncomp=self.tree_local.num_cells() * self.ncomp, dtype=REAL)
        self.tree_local_ga_offsets = data.ScalarArray(ncomp=self.R+1, dtype=INT64)
        
        
        t = 0
        self.tree_local_ga_offsets[0] = 0
        for rx in range(self.R):
            t += self.tree_local.num_data[rx]
            self.tree_local_ga_offsets[rx + 1] = t
        
        s = self.subdivision
        s = [sx ** (self.R - 1) for sx in s]
        ncells_finest = s[0] * s[1] * s[2]
        self.cell_occupation_ga = data.GlobalArray(ncomp=ncells_finest, dtype=INT64)

        
        # interaction lists
        self.il = fmm_interaction_lists.compute_interaction_lists(domain.extent, self.subdivision)
        self.il_max_len = max(len(lx) for lx in self.il[0])
        
        self.il_array = np.array(self.il[0], INT64)
        self.il_scalararray = data.ScalarArray(ncomp=self.il_array.size, dtype=INT64)
        self.il_scalararray[:] = self.il_array.ravel().copy()

        
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
        
        tmp = 0
        for datx in g.particle_dats:
            tmp += getattr(g, datx)._dat.nbytes
        self.dat_size = tmp


        self.direct = DirectCommon(positions, charges, domain, boundary_condition, r, self.subdivision, g._mc_fmm_cells, g._mc_ids)

        self.energy = None

        self._cell_bin_loop = None
        self._init_bin_loop()

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


                inline_local_exp(
                    Q.i[0], 
                    sqrt(xy2 + dz * dz),
                    atan2(sqrt(xy2), dz),
                    atan2(dy, dx),
                    &TL[TL_OFFSETS[MC_LEVEL.i[ox]] + NCOMP * MC_CL.i[ox]]
                );


            }}

            '''.format(
            ),
            (   
                Constant('R', self.R),
                Constant('NCOMP', self.ncomp),
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
            ),
            headers=(
                lib.build.write_header(
                    self.mc_lee.create_local_exp_header + \
                    self.mc_lee.create_local_exp_src
                ),
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
            'TL_OFFSETS': self.tree_local_ga_offsets(access.READ),
            'TL': self.tree_local_ga(access.INC_ZERO),
        }

        self._cell_bin_loop = loop.ParticleLoopOMP(kernel=k, dat_dict=dat_dict)



    
    
    def _setup_tree(self):

        self._cell_bin_loop.execute()
        self._profile_inc('_cell_bin_wrapper', self._cell_bin_loop.wrapper_timer.time())

        for rx in range(self.R):
            
            s = self.tree_local_ga_offsets[rx]
            e = self.tree_local_ga_offsets[rx+1]
            self.tree_local[rx].ravel()[:] = self.tree_local_ga[s:e:].copy()

    
    def _compute_energy(self):
        

        energy = 0.0
        energy_py = 0.0
        for px in range(self.group.npart_local):
            tmp = self._get_old_energy(px)
            energy_py += tmp

        energy += energy_py

        return 0.5 * energy
    

    def _get_old_energy(self, px):

        g = self.positions.group
        C = g._mc_fmm_cells

        energy = 0.0

        pos = self.positions[px,:].copy()

        charge = float(self.charges[px, 0])

        # cell on finest level
        cell = (C[px, 0], C[px, 1], C[px, 2])
        
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

        direct_contrib = self.direct.get_old_energy(px)
        energy += direct_contrib

        return energy


    def _get_new_energy(self, px, pos):

        energy = 0.0
        charge = float(self.charges[px, 0])

        cell = self._get_cell(pos)

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

        direct_contrib = self.direct.get_new_energy(px, pos)

        energy += direct_contrib


        return energy


    def initialise(self):
        N = self.positions.npart_local
        g = self.positions.group
        with g._mc_ids.modify_view() as mv:
            mv[:, 0] = np.arange(N)

        self._setup_tree()
        self.direct.initialise()
        self.energy = self._compute_energy()

    
    def propose(self, move):
        px = int(move[0])
        pos = move[1]
        
        old_energy = self._get_old_energy(px)
        new_energy = self._get_new_energy(px, pos)
        self_energy = self._get_self_interaction(px, pos)
        #print("L\t-->", old_energy, new_energy, self_energy)

        self._profile_inc('num_propose', 1)
        return  new_energy - old_energy - self_energy


    def accept(self, move, energy_diff=None):
        px = int(move[0])
        new_pos = move[1]

        new_cell = self._get_cell(new_pos)
        g = self.positions.group

        if energy_diff is None:
            energy_diff = self.propose(move)
        
        self.energy += energy_diff


        old_pos = self.positions[px, :].copy()
        q = float(self.charges[px, 0])
        

        t0 = time.time()
        self.direct.accept(move)
        self._profile_inc('direct_accept', time.time() - t0)       

        new_position = np.array((new_pos[0], new_pos[1], new_pos[2]), REAL)
        old_position = np.array((old_pos[0], old_pos[1], old_pos[2]), REAL)


        t0 = time.time()
        self._indirect_accept_lib(
            old_position.ctypes.get_as_parameter(),
            new_position.ctypes.get_as_parameter(),
            REAL(q),
            self.il_array.ctypes.get_as_parameter(),
            self.tree_local_ptrs.ctypes.get_as_parameter()
        )
        self._profile_inc('indirect_accept', time.time() - t0)



        t0 = time.time()
        assert self.comm.size == 1
        with g._mc_fmm_cells.modify_view() as m:
            m[px, :] = new_cell

        
        # move the particle in the dats
        with self.positions.modify_view() as m:
            m[px, :] = new_pos.copy()

        self._profile_inc('dats_accept', time.time() - t0)
        self._profile_inc('num_accept', 1)



