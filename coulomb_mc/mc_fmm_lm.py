

import numpy as np
from ppmd import data, loop, kernel, access, lib, opt, pairloop
from ppmd.coulomb import fmm_interaction_lists, octal, lm
from coulomb_kmc import kmc_expansion_tools, common
import ctypes
import math

from coulomb_mc import mc_expansion_tools
import time


from ppmd.coulomb.sph_harm import *
from coulomb_mc.mc_direct import DirectCommon
from coulomb_mc.mc_common import MCCommon, BCType



Constant = kernel.Constant
REAL = ctypes.c_double
INT64 = ctypes.c_int64

PROFILE = opt.PROFILE

class MCFMM_LM(MCCommon):

    def __init__(self, positions, charges, domain, boundary_condition, r, l):

        assert boundary_condition in ('free_space', 'pbc')

        self.positions = positions
        self.charges = charges
        self.domain = domain
        self.comm = self.domain.comm
        self.boundary_condition = BCType(boundary_condition)
        self.R = r
        self.L = l
        self.ncomp = (self.L ** 2) * 2
        self.group = self.positions.group

        self.lm = lm.PyLM(positions, charges, domain, boundary_condition, r, l)
        self.sph_gen = self.lm.sph_gen
        self.solver = self.lm

        self.energy = None

        pd = type(self.charges)
        self.group._mc_lm_ids = pd(ncomp=1, dtype=INT64)
        self.direct = DirectCommon(positions, charges, domain, boundary_condition,
            r, self.lm.subdivision, self.group._lm_fine_cells, self.group._mc_lm_ids)
        
        self.sh = pairloop.state_handler.StateHandler(state=None, shell_cutoff=self.lm.max_cell_width, pair=False)

        self._init_libs()

        self.tree = None
        
        if self.boundary_condition == BCType.PBC:
            self.lrc = self.lm.lrc
            self._init_lr_correction_libs()

        self._init_single_propose_lib()


    def initialise(self):
        N = self.positions.npart_local
        g = self.positions.group
        with g._mc_lm_ids.modify_view() as mv:
            mv[:, 0] = np.arange(N)
        
        t0 = time.time()
        self.energy = self.lm(self.positions, self.charges)
        self._profile_inc('initialise_solve', time.time() - t0)

        t0 = time.time()
        self.direct.initialise()
        self._profile_inc('initialise_direct_initialise', time.time() - t0)
        self.tree = self.lm.tree[:].copy()
        
        if self.boundary_condition == BCType.PBC:
            self.mvector = self.solver.mvector.copy()
            self.evector = self.solver.evector.copy()
            self.lr_energy = self.solver.lr_energy

    def _inc_prop_count(self):
        self._profile_inc('num_propose', 1)


    def _update_profiling(self, old_time_direct, old_time_indirect, new_time_direct, new_time_indirect):

        self._profile_inc('direct_new_inner', new_time_direct)
        self._profile_inc('direct_old_inner', old_time_direct)
        self._profile_inc('indirect_new_inner', new_time_indirect)
        self._profile_inc('indirect_old_inner', old_time_indirect)
    
    def accept(self, move, energy_diff=None):
        

        px = int(move[0])
        new_pos = move[1]

        new_cell, mm_cells, mm_child_index = self._get_new_cells(new_pos)

        g = self.positions.group

        if energy_diff is None:
            energy_diff = self.propose(move)
        
        self.energy += energy_diff

        t0 = time.time()
        self._lr_accept(px, new_pos)
        self._profile_inc('lm_accept', time.time() - t0)


        t0 = time.time()
        self.direct.accept(move)
        self._profile_inc('direct_accept', time.time() - t0)       

        new_pos = np.array(
            (new_pos[0], new_pos[1], new_pos[2]), 
            REAL
        )
        
        new_cell = np.array(
            (new_cell[0], new_cell[1], new_cell[2]), 
            INT64
        )

        t0 = time.time()
        
        time_taken = REAL(0)

        self._indirect_accept_lib(
            INT64(px),
            self.sh.get_pointer(self.positions(access.READ)),
            self.sh.get_pointer(self.charges(access.READ)),
            self.sh.get_pointer(self.group._lm_cells(access.READ)),
            self.tree.ctypes.get_as_parameter(),
            self.lm.il_scalararray.ctypes_data,
            new_pos.ctypes.get_as_parameter(),
            mm_cells.ctypes.get_as_parameter(),
            ctypes.byref(time_taken)
        )
        self._profile_inc('indirect_accept', time.time() - t0)
        self._profile_inc('indirect_accept_internal', time_taken.value)



        t0 = time.time()
        assert self.comm.size == 1



        # update the finest level cells
        with self.group._lm_fine_cells.modify_view() as m:
            m[px, :] = new_cell
        
        # update the cells on each level
        with self.group._lm_cells.modify_view() as m:
            m[px, :] = mm_cells.ravel()

        # update the child indices on each level
        with self.group._lm_child_index.modify_view() as m:
            m[px, :] = mm_child_index.ravel()

        
        # move the particle in the dats
        with self.positions.modify_view() as m:
            m[px, :] = new_pos.copy()

        self._profile_inc('dats_accept', time.time() - t0)
        self._profile_inc('num_accept', 1)
    

    def free(self):
        pass


    def _get_new_cells(self, position):

        mm_cells = np.zeros((self.R, 3), INT64)
        mm_child_index = np.zeros((self.R, 3), INT64)

        e = self.domain.extent

        sr0 = position[0] + 0.5 * e[0]
        sr1 = position[1] + 0.5 * e[1]
        sr2 = position[2] + 0.5 * e[2]
        cell_widths = [1.0 / (ex / (sx**(self.R - 1))) for ex, sx in zip(e, self.lm.subdivision)]
        c0 = int(sr0 * cell_widths[0])
        c1 = int(sr1 * cell_widths[1])
        c2 = int(sr2 * cell_widths[2])
        sd0 = self.lm.subdivision[0]
        sd1 = self.lm.subdivision[1]
        sd2 = self.lm.subdivision[2]
        sl0 = sd0 ** (self.R-1)
        sl1 = sd1 ** (self.R-1)
        sl2 = sd2 ** (self.R-1)
        c0 = 0 if c0 < 0 else ((sl0-1) if c0 >= sl0 else c0)
        c1 = 0 if c1 < 0 else ((sl1-1) if c1 >= sl1 else c1)
        c2 = 0 if c2 < 0 else ((sl2-1) if c2 >= sl2 else c2)

        for rx in range(self.R-1, -1, -1):
            mm_cells[rx, :] = (c0, c1, c2)
            mm_child_index[rx, :] = (
                c0 % sd0,
                c1 % sd1,
                c2 % sd2
            )
            c0 //= sd0
            c1 //= sd1
            c2 //= sd2

        new_cell = mm_cells[self.R-1, :].copy()

        return new_cell, mm_cells, mm_child_index




    def _get_new_energy(self, ix, position, charge):


        ie = REAL(0)
        
        position = np.array((position[0], position[1], position[2]), REAL)
        charge = REAL(charge)
        
        new_cell, mm_cells, mm_child_index = self._get_new_cells(position)
        
        time_taken = REAL(0)
        
        t0 = time.time()
        self._indirect_lib(
            INT64(0),
            position.ctypes.get_as_parameter(),
            ctypes.byref(charge),
            mm_cells.ctypes.get_as_parameter(),
            mm_child_index.ctypes.get_as_parameter(),
            self.tree.ctypes.get_as_parameter(),
            ctypes.byref(ie),
            ctypes.byref(time_taken)
        )    
        self._profile_inc('indirect_get_new', time.time() - t0)
        self._profile_inc('indirect_get_new_internal', time_taken.value)
        
        t0 = time.time()
        direct_contrib = self.direct.get_new_energy(ix, position)
        self._profile_inc('direct_get_new', time.time() - t0)

        #print("L GET NEW", "direct:", direct_contrib, "indirect:", ie.value)

        return ie.value + direct_contrib



    def _get_old_energy(self, ix):
        
        ie = REAL(0)
        time_taken = REAL(0)
        
        t0 = time.time()
        self._indirect_lib(
            INT64(ix),
            self.sh.get_pointer(self.positions(access.READ)),
            self.sh.get_pointer(self.charges(access.READ)),
            self.sh.get_pointer(self.group._lm_cells(access.READ)),
            self.sh.get_pointer(self.group._lm_child_index(access.READ)),
            self.tree.ctypes.get_as_parameter(),
            ctypes.byref(ie),
            ctypes.byref(time_taken)
        )

        self._profile_inc('indirect_get_old', time.time() - t0)
        self._profile_inc('indirect_get_old_internal', time_taken.value)


        t0 = time.time()
        direct_contrib = self.direct.get_old_energy(ix)
        self._profile_inc('direct_get_old', time.time() - t0)

        #print("L GET OLD", "direct:", direct_contrib, "indirect:", ie.value)
        return ie.value + direct_contrib




    def _init_libs(self):

        bc = self.boundary_condition

        if bc == BCType.FREE_SPACE:
            bc_block = r'''
                if (ocx < 0) {{continue;}}
                if (ocy < 0) {{continue;}}
                if (ocz < 0) {{continue;}}
                if (ocx >= ncx) {{continue;}}
                if (ocy >= ncx) {{continue;}}
                if (ocz >= ncx) {{continue;}}
            '''

        elif bc in (BCType.PBC, BCType.NEAREST):
            bc_block = r'''
                ocx = (ocx + ncx) % ncx;
                ocy = (ocy + ncy) % ncy;
                ocz = (ocz + ncz) % ncz;
            '''
        else:
            raise RuntimeError('Unknown or not implemented boundary condition.')
        

        g = self.group
        extent = self.domain.extent
        cell_widths = [1.0 / (ex / (sx**(self.R - 1))) for ex, sx in zip(extent, self.lm.subdivision)]
        
        L = self.L

        sph_gen = self.lm.sph_gen

        def cube_ind(L, M):
            return ((L) * ( (L) + 1 ) + (M) )


        EC = ''
        for lx in range(L):

            for mx in range(-lx, lx+1):
                smx = 'n' if mx < 0 else 'p'
                smx += str(abs(mx))

                re_lnm = SphSymbol('reln{lx}m{mx}'.format(lx=lx, mx=smx))
                im_lnm = SphSymbol('imln{lx}m{mx}'.format(lx=lx, mx=smx))

                EC += '''
                const double {re_lnm} = TREE[OFFSET + {cx}];
                const double {im_lnm} = TREE[OFFSET + IM_OFFSET + {cx}];
                '''.format(
                    re_lnm=str(re_lnm),
                    im_lnm=str(im_lnm),
                    cx=str(cube_ind(lx, mx))
                )
                cm_re, cm_im = cmplx_mul(re_lnm, im_lnm, sph_gen.get_y_sym(lx, mx)[0],
                    sph_gen.get_y_sym(lx, mx)[1])
                EC += 'tmp_energy += ({cm_re}) * rhol;\n'.format(cm_re=cm_re)
                
            EC += 'rhol *= radius;\n'


        src = r'''
        
        #include <math.h>
        #include <chrono>
        #include <stdio.h>

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
    
        namespace INDIRECT {{

        const REAL WIDTHS_X[R] = {{ {WIDTHS_X} }};
        const REAL WIDTHS_Y[R] = {{ {WIDTHS_Y} }};
        const REAL WIDTHS_Z[R] = {{ {WIDTHS_Z} }};

        const INT64 NCELLS_X[R] = {{ {NCELLS_X} }};
        const INT64 NCELLS_Y[R] = {{ {NCELLS_Y} }};
        const INT64 NCELLS_Z[R] = {{ {NCELLS_Z} }};

        const INT64 LEVEL_OFFSETS[R] = {{ {LEVEL_OFFSETS} }};


        static inline void kernel(
            const REAL  * RESTRICT P,
            const REAL  * RESTRICT Q,
            const INT64 * RESTRICT MM_CELLS,
            const INT64 * RESTRICT MM_CHILD_INDEX,
            const REAL  * RESTRICT TREE,
                  REAL  * RESTRICT OUT_ENERGY
        ){{
            
            const double rx = P[0];
            const double ry = P[1];
            const double rz = P[2];

            double particle_energy = 0.0;

            for( int level=1 ; level<R ; level++ ){{

                // cell on this level
                const int64_t cfx = MM_CELLS[level*3 + 0];
                const int64_t cfy = MM_CELLS[level*3 + 1];
                const int64_t cfz = MM_CELLS[level*3 + 2];

                const double wx = WIDTHS_X[level];
                const double wy = WIDTHS_Y[level];
                const double wz = WIDTHS_Z[level];
                
            
                const int64_t lin_ind = cfx + NCELLS_X[level] * (cfy + NCELLS_Y[level] * cfz);

                const double dx = rx - ((-HEX) + (0.5 * wx) + (cfx * wx));
                const double dy = ry - ((-HEY) + (0.5 * wy) + (cfy * wy));
                const double dz = rz - ((-HEZ) + (0.5 * wz) + (cfz * wz));

                const double xy2 = dx * dx + dy * dy;
                const double radius = sqrt(xy2 + dz * dz);
                const double theta = atan2(sqrt(xy2), dz);
                const double phi = atan2(dy, dx);
                
                const int64_t OFFSET = LEVEL_OFFSETS[level] + NCOMP * lin_ind;

                {SPH_GEN}
                double rhol = 1.0;
                double tmp_energy = 0.0;
                {ENERGY_COMP}

                //printf("    L R = %d contrib = %f\n", level, tmp_energy);

                particle_energy += tmp_energy;

            }}


            OUT_ENERGY[0] = particle_energy * Q[0];

        }}



        int get_energy(
            const INT64            IX,
            const REAL  * RESTRICT P,
            const REAL  * RESTRICT Q,
            const INT64 * RESTRICT MM_CELLS,
            const INT64 * RESTRICT MM_CHILD_INDEX,
            const REAL  * RESTRICT TREE,
                  REAL  * RESTRICT OUT_ENERGY,
                  REAL  * RESTRICT TIME_TAKEN
        ){{
            
            std::chrono::high_resolution_clock::time_point _loop_timer_t0 = std::chrono::high_resolution_clock::now();
            
            kernel(
                &P[IX*3],
                &Q[IX],
                &MM_CELLS[IX*THREE_R],
                &MM_CHILD_INDEX[IX*THREE_R],
                TREE,
                OUT_ENERGY
            );

            std::chrono::high_resolution_clock::time_point _loop_timer_t1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> _loop_timer_res = _loop_timer_t1 - _loop_timer_t0;
            *TIME_TAKEN = (double) _loop_timer_res.count();
                
            return 0;
        }}

        }}


        extern "C" int get_indirect_energy(
            const INT64            IX,
            const REAL  * RESTRICT P,
            const REAL  * RESTRICT Q,
            const INT64 * RESTRICT MM_CELLS,
            const INT64 * RESTRICT MM_CHILD_INDEX,
            const REAL  * RESTRICT TREE,
                  REAL  * RESTRICT OUT_ENERGY,
                  REAL  * RESTRICT TIME_TAKEN
        ){{


        return INDIRECT::get_energy(
            IX,
            P,
            Q,
            MM_CELLS,
            MM_CHILD_INDEX,
            TREE,
            OUT_ENERGY,
            TIME_TAKEN
        );

        }}

        #undef R                 
        #undef EX                
        #undef EY                
        #undef EZ                
        #undef HEX               
        #undef HEY               
        #undef HEZ               
        #undef CWX               
        #undef CWY               
        #undef CWZ               
        #undef LCX               
        #undef LCY               
        #undef LCZ               
        #undef SDX               
        #undef SDY               
        #undef SDZ               
        #undef IL_NO             
        #undef IL_STRIDE_OUTER   
        #undef NCOMP             
        #undef IM_OFFSET         
        #undef THREE_R

        '''.format(
            SPH_GEN=str(sph_gen.module),
            ENERGY_COMP=str(EC),
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
            LCX=self.lm.subdivision[0] ** (self.R - 1),
            LCY=self.lm.subdivision[1] ** (self.R - 1),
            LCZ=self.lm.subdivision[2] ** (self.R - 1),
            SDX=self.lm.subdivision[0],
            SDY=self.lm.subdivision[1],
            SDZ=self.lm.subdivision[2],
            IL_NO=self.lm.il_array.shape[1],
            IL_STRIDE_OUTER=self.lm.il_array.shape[1] * self.lm.il_array.shape[2],
            NCOMP=self.ncomp,
            IM_OFFSET=self.L**2,
            THREE_R=self.R*3,
            WIDTHS_X=self.lm.widths_x_str,
            WIDTHS_Y=self.lm.widths_y_str,
            WIDTHS_Z=self.lm.widths_z_str,
            NCELLS_X=self.lm.ncells_x_str,
            NCELLS_Y=self.lm.ncells_y_str,
            NCELLS_Z=self.lm.ncells_z_str,
            LEVEL_OFFSETS=self.lm.level_offsets_str
        )

        self._indirect_lib = lib.build.simple_lib_creator('', src)['get_indirect_energy']
        

        # =============================================================================================================


        self.lib_source = src

        self.lib_parameters = [
            'const REAL  * RESTRICT INDIRECT_TREE',
            'const INT64            INDIRECT_OLD_IX',
            'const REAL  * RESTRICT INDIRECT_OLD_P',
            'const REAL  * RESTRICT INDIRECT_OLD_Q',
            'const INT64 * RESTRICT INDIRECT_OLD_MM_CELLS',
            'const INT64 * RESTRICT INDIRECT_OLD_MM_CHILD_INDEX',
            '      REAL  * RESTRICT INDIRECT_OLD_OUT_ENERGY',
            '      REAL  * RESTRICT INDIRECT_OLD_TIME_TAKEN',
            'const INT64            INDIRECT_NEW_IX',
            'const REAL  * RESTRICT INDIRECT_NEW_P',
            'const REAL  * RESTRICT INDIRECT_NEW_Q',
            'const INT64 * RESTRICT INDIRECT_NEW_MM_CELLS',
            'const INT64 * RESTRICT INDIRECT_NEW_MM_CHILD_INDEX',
            '      REAL  * RESTRICT INDIRECT_NEW_OUT_ENERGY',
            '      REAL  * RESTRICT INDIRECT_NEW_TIME_TAKEN',
        ]

        
        self.lib_call_old = '''
        INDIRECT::get_energy(
            INDIRECT_OLD_IX,
            INDIRECT_OLD_P,
            INDIRECT_OLD_Q,
            INDIRECT_OLD_MM_CELLS,
            INDIRECT_OLD_MM_CHILD_INDEX,
            INDIRECT_TREE,
            INDIRECT_OLD_OUT_ENERGY,
            INDIRECT_OLD_TIME_TAKEN
        );
        '''

        self.lib_call_new = '''
        INDIRECT::get_energy(
            INDIRECT_NEW_IX,
            INDIRECT_NEW_P,
            INDIRECT_NEW_Q,
            INDIRECT_NEW_MM_CELLS,
            INDIRECT_NEW_MM_CHILD_INDEX,
            INDIRECT_TREE,
            INDIRECT_NEW_OUT_ENERGY,
            INDIRECT_NEW_TIME_TAKEN
        );
        '''

        # =============================================================================================================




        assign_gen =  'const double iradius = 1.0 / radius;\n'
        assign_gen += 'double rholcharge = iradius * charge;\n'
        for lx in range(self.L):
            for mx in range(-lx, lx+1):
                assign_gen += 'TREE[OFFSET + {ind}] += {ylmm} * rholcharge;\n'.format(
                        ind=cube_ind(lx, mx),
                        ylmm=str(sph_gen.get_y_sym(lx, -mx)[0])
                    )
                assign_gen += 'TREE[OFFSET + IM_OFFSET + {ind}] += {ylmm} * rholcharge;\n'.format(
                        ind=cube_ind(lx, mx),
                        ylmm=str(sph_gen.get_y_sym(lx, -mx)[1])
                    )
            assign_gen += 'rholcharge *= iradius;\n'


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

        const REAL WIDTHS_X[R] = {{ {WIDTHS_X} }};
        const REAL WIDTHS_Y[R] = {{ {WIDTHS_Y} }};
        const REAL WIDTHS_Z[R] = {{ {WIDTHS_Z} }};

        const INT64 NCELLS_X[R] = {{ {NCELLS_X} }};
        const INT64 NCELLS_Y[R] = {{ {NCELLS_Y} }};
        const INT64 NCELLS_Z[R] = {{ {NCELLS_Z} }};

        const INT64 LEVEL_OFFSETS[R] = {{ {LEVEL_OFFSETS} }};


        static inline void kernel(
            const REAL  * RESTRICT P,
            const REAL  * RESTRICT Q,
            const INT64 * RESTRICT MM_CELLS,
                  REAL  * RESTRICT TREE,
            const INT64 * RESTRICT IL
        ){{

            const double rx = P[0];
            const double ry = P[1];
            const double rz = P[2];
            const double charge = Q[0];


            // compute the local expansions
            #pragma omp parallel for collapse(2)
            for( int level=1 ; level<R ; level++) {{
                 // loop over IL for this child cell
                for( int ox=0 ; ox<IL_NO ; ox++){{

                    // cell on this level
                    const int64_t cfx = MM_CELLS[level * 3 + 0];
                    const int64_t cfy = MM_CELLS[level * 3 + 1];
                    const int64_t cfz = MM_CELLS[level * 3 + 2];

                    // child on this level
                    const int64_t cix = cfx % SDX;
                    const int64_t ciy = cfy % SDY;
                    const int64_t ciz = cfz % SDZ;
                    const int64_t ci = cix + SDX * (ciy + SDY * ciz);

                    // cell widths on this level
                    const double wx = WIDTHS_X[level];
                    const double wy = WIDTHS_Y[level];
                    const double wz = WIDTHS_Z[level];

                    // number of cells on this level
                    const int64_t ncx = NCELLS_X[level];
                    const int64_t ncy = NCELLS_Y[level];
                    const int64_t ncz = NCELLS_Z[level];



                    int64_t ocx = cfx + IL[ci * IL_STRIDE_OUTER + ox * 3 + 0];
                    int64_t ocy = cfy + IL[ci * IL_STRIDE_OUTER + ox * 3 + 1];
                    int64_t ocz = cfz + IL[ci * IL_STRIDE_OUTER + ox * 3 + 2];

                    const double dx = rx - ((-HEX) + (0.5 * wx) + (ocx * wx));
                    const double dy = ry - ((-HEY) + (0.5 * wy) + (ocy * wy));
                    const double dz = rz - ((-HEZ) + (0.5 * wz) + (ocz * wz));

                    {BC_BLOCK}

                    const int64_t lin_ind = ocx + NCELLS_X[level] * (ocy + NCELLS_Y[level] * ocz);



                    const double xy2 = dx * dx + dy * dy;
                    const double radius = sqrt(xy2 + dz * dz);
                    const double theta = atan2(sqrt(xy2), dz);
                    const double phi = atan2(dy, dx);
                    
                    const int64_t OFFSET = LEVEL_OFFSETS[level] + NCOMP * lin_ind;

                    {SPH_GEN}
                    {ASSIGN_GEN}

                }}
            }}

            
        }}



        extern "C" int indirect_accept(
            const INT64            IX,
            const REAL  * RESTRICT P,
            const REAL  * RESTRICT Q,
            const INT64 * RESTRICT MM_CELLS,
                  REAL  * RESTRICT TREE,
            const INT64 * RESTRICT IL,
            const REAL  * RESTRICT NEW_P,
            const INT64 * RESTRICT NEW_MM_CELLS,
                  REAL  * RESTRICT TIME_TAKEN
        ){{
            
            // remove the old contrib

            std::chrono::high_resolution_clock::time_point _loop_timer_t0 = std::chrono::high_resolution_clock::now();
            
            const REAL tmp_q = -1.0 * Q[IX];
            kernel(
                &P[IX*3],
                &tmp_q,
                &MM_CELLS[IX*THREE_R],
                TREE,
                IL
            );

            
            // add the new contrib
            kernel(
                NEW_P,
                &Q[IX],
                NEW_MM_CELLS,
                TREE,
                IL
            );

 
            std::chrono::high_resolution_clock::time_point _loop_timer_t1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> _loop_timer_res = _loop_timer_t1 - _loop_timer_t0;
            *TIME_TAKEN = (double) _loop_timer_res.count();
               
            return 0;
        }}

        '''.format(
            BC_BLOCK=bc_block,
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
            LCX=self.lm.subdivision[0] ** (self.R - 1),
            LCY=self.lm.subdivision[1] ** (self.R - 1),
            LCZ=self.lm.subdivision[2] ** (self.R - 1),
            SDX=self.lm.subdivision[0],
            SDY=self.lm.subdivision[1],
            SDZ=self.lm.subdivision[2],
            IL_NO=self.lm.il_array.shape[1],
            IL_STRIDE_OUTER=self.lm.il_array.shape[1] * self.lm.il_array.shape[2],
            NCOMP=self.ncomp,
            IM_OFFSET=self.L**2,
            THREE_R=self.R*3,
            WIDTHS_X=self.lm.widths_x_str,
            WIDTHS_Y=self.lm.widths_y_str,
            WIDTHS_Z=self.lm.widths_z_str,
            NCELLS_X=self.lm.ncells_x_str,
            NCELLS_Y=self.lm.ncells_y_str,
            NCELLS_Z=self.lm.ncells_z_str,
            LEVEL_OFFSETS=self.lm.level_offsets_str
        )

        self._indirect_accept_lib = lib.build.simple_lib_creator(
            '#include <math.h>\n#include <chrono>\n#include <stdio.h>', 
            src
        )['indirect_accept']



    def get_lib_combined_args(self, ix, position, old_energy, new_energy, old_time_taken, new_time_taken):


        ie = REAL(0)
        
        position = np.array((position[0], position[1], position[2]), REAL)
        charge = REAL(self.charges[ix,0])
        
        new_cell, mm_cells, mm_child_index = self._get_new_cells(position)
        

        args = [
            self.tree.ctypes.get_as_parameter(),
            INT64(ix),
            self.sh.get_pointer(self.positions(access.READ)),
            self.sh.get_pointer(self.charges(access.READ)),
            self.sh.get_pointer(self.group._lm_cells(access.READ)),
            self.sh.get_pointer(self.group._lm_child_index(access.READ)),
            ctypes.byref(old_energy),
            ctypes.byref(old_time_taken),
            INT64(0),
            position.ctypes.get_as_parameter(),
            ctypes.byref(charge),
            mm_cells.ctypes.get_as_parameter(),
            mm_child_index.ctypes.get_as_parameter(),
            ctypes.byref(new_energy),
            ctypes.byref(new_time_taken)
        ]

        assert len(args) == len(self.lib_parameters)
        return args, (new_cell, mm_cells, mm_child_index)


