

import numpy as np
from ppmd import data, loop, kernel, access, lib, opt, pairloop
from ppmd.coulomb import fmm_interaction_lists, octal, mm
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

class MCFMM_MM(MCCommon):

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

        self.mm = mm.PyMM(positions, charges, domain, boundary_condition, r, l)
        self.solver = self.mm

        self.sph_gen = self.mm.sph_gen

        self.energy = None

        pd = type(self.charges)
        self.group._mc_mm_ids = pd(ncomp=1, dtype=INT64)
        self.direct = DirectCommon(positions, charges, domain, boundary_condition,
            r, self.mm.subdivision, self.group._mm_fine_cells, self.group._mc_mm_ids)
        
        self.sh = pairloop.state_handler.StateHandler(state=None, shell_cutoff=self.mm.max_cell_width, pair=False)

        self._init_libs()

        self.tree = None

        if self.boundary_condition == BCType.PBC:
            self.lrc = self.mm.lrc
            self._init_lr_correction_libs()


    def initialise(self):
        N = self.positions.npart_local
        g = self.positions.group
        with g._mc_mm_ids.modify_view() as mv:
            mv[:, 0] = np.arange(N)

        self.energy = self.mm(self.positions, self.charges)
        self.direct.initialise()
        self.tree = self.mm.tree[:].copy()

        if self.boundary_condition == BCType.PBC:
            self.mvector = self.solver.mvector.copy()
            self.evector = self.solver.evector.copy()
            self.lr_energy = self.solver.lr_energy


    def propose(self, move):
        px = int(move[0])
        pos = move[1]

        old_energy = self._get_old_energy(px)
        new_energy = self._get_new_energy(px, pos, self.charges[px, 0])
        self_energy = self._get_self_interaction(px, pos)
        #print("M\t-->", old_energy, new_energy, self_energy)

        self._profile_inc('num_propose', 1)
        return new_energy - old_energy - self_energy

    
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
        self._profile_inc('lr_accept', time.time() - t0)


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
            self.sh.get_pointer(self.group._mm_cells(access.READ)),
            self.tree.ctypes.get_as_parameter(),
            new_pos.ctypes.get_as_parameter(),
            mm_cells.ctypes.get_as_parameter(),
            ctypes.byref(time_taken)
        )
        self._profile_inc('indirect_accept', time.time() - t0)
        self._profile_inc('indirect_accept_internal', time_taken.value)



        t0 = time.time()
        assert self.comm.size == 1



        # update the finest level cells
        with self.group._mm_fine_cells.modify_view() as m:
            m[px, :] = new_cell
        
        # update the cells on each level
        with self.group._mm_cells.modify_view() as m:
            m[px, :] = mm_cells.ravel()

        # update the child indices on each level
        with self.group._mm_child_index.modify_view() as m:
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
        cell_widths = [1.0 / (ex / (sx**(self.R - 1))) for ex, sx in zip(e, self.mm.subdivision)]
        c0 = int(sr0 * cell_widths[0])
        c1 = int(sr1 * cell_widths[1])
        c2 = int(sr2 * cell_widths[2])
        sd0 = self.mm.subdivision[0]
        sd1 = self.mm.subdivision[1]
        sd2 = self.mm.subdivision[2]
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
            self.mm.il_scalararray.ctypes_data,
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


        return ie.value + direct_contrib



    def _get_old_energy(self, ix):
        
        ie = REAL(0)
        time_taken = REAL(0)
        
        t0 = time.time()
        self._indirect_lib(
            INT64(ix),
            self.sh.get_pointer(self.positions(access.READ)),
            self.sh.get_pointer(self.charges(access.READ)),
            self.mm.il_scalararray.ctypes_data,
            self.sh.get_pointer(self.group._mm_cells(access.READ)),
            self.sh.get_pointer(self.group._mm_child_index(access.READ)),
            self.tree.ctypes.get_as_parameter(),
            ctypes.byref(ie),
            ctypes.byref(time_taken)
        )

        self._profile_inc('indirect_get_old', time.time() - t0)
        self._profile_inc('indirect_get_old_internal', time_taken.value)


        t0 = time.time()
        direct_contrib = self.direct.get_old_energy(ix)
        self._profile_inc('direct_get_old', time.time() - t0)

        #print("M GET OLD", "direct:", direct_contrib, "indirect:", ie.value)
        return ie.value + direct_contrib




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


        widths_x = ','.join([str(ix) for ix in  self.mm.widths_x])
        widths_y = ','.join([str(ix) for ix in  self.mm.widths_y])
        widths_z = ','.join([str(ix) for ix in  self.mm.widths_z])
        ncells_x = ','.join([str(ix) for ix in  self.mm.ncells_x])
        ncells_y = ','.join([str(ix) for ix in  self.mm.ncells_y])
        ncells_z = ','.join([str(ix) for ix in  self.mm.ncells_z])
        
        level_offsets = [0]
        nx = 1
        ny = 1
        nz = 1
        for level in range(1, self.R):
            level_offsets.append(
                level_offsets[-1] + nx * ny * nz * self.ncomp
            )
            nx *= self.mm.subdivision[0]
            ny *= self.mm.subdivision[1]
            nz *= self.mm.subdivision[2]

        level_offsets = ','.join([str(ix) for ix in level_offsets])


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
            const INT64 * RESTRICT IL,
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
                  REAL  * RESTRICT OUT_ENERGY,
                  REAL  * RESTRICT TIME_TAKEN
        ){{
            
            std::chrono::high_resolution_clock::time_point _loop_timer_t0 = std::chrono::high_resolution_clock::now();
            
            kernel(
                &P[IX*3],
                &Q[IX],
                IL,
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
            THREE_R=self.R*3,
            WIDTHS_X=widths_x,
            WIDTHS_Y=widths_y,
            WIDTHS_Z=widths_z,
            NCELLS_X=ncells_x,
            NCELLS_Y=ncells_y,
            NCELLS_Z=ncells_z,
            LEVEL_OFFSETS=level_offsets            
        )

        self._indirect_lib = lib.build.simple_lib_creator('#include <math.h>\n#include <chrono>', src)['get_energy']
        

        # =============================================================================================================


        assign_gen =  'double rhol = 1.0;\n'
        assign_gen += 'double rholcharge = rhol * charge;\n'
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
            assign_gen += 'rhol *= radius;\n'
            assign_gen += 'rholcharge = rhol * charge;\n'








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
                  REAL  * RESTRICT TREE
        ){{

            const double rx = P[0];
            const double ry = P[1];
            const double rz = P[2];


            for( int level=R-1 ; level>=0 ; level-- ){{

                const INT64 cellx = MM_CELLS[level*3 + 0];
                const INT64 celly = MM_CELLS[level*3 + 1];
                const INT64 cellz = MM_CELLS[level*3 + 2];


                const double dx = rx - ((-HEX) + (0.5  + cellx) * WIDTHS_X[level]);
                const double dy = ry - ((-HEY) + (0.5  + celly) * WIDTHS_Y[level]);
                const double dz = rz - ((-HEZ) + (0.5  + cellz) * WIDTHS_Z[level]);

                const double xy2 = dx * dx + dy * dy;
                const double radius = sqrt(xy2 + dz * dz);
                const double theta = atan2(sqrt(xy2), dz);
                const double phi = atan2(dy, dx);
                const double charge = Q[0];
                
                const int64_t lin_ind = cellx + NCELLS_X[level] * (celly + NCELLS_Y[level] * cellz);
                const int64_t OFFSET = LEVEL_OFFSETS[level] + NCOMP * lin_ind;

                {SPH_GEN}
                {ASSIGN_GEN}

            }}

            
        }}



        extern "C" int indirect_accept(
            const INT64            IX,
            const REAL  * RESTRICT P,
            const REAL  * RESTRICT Q,
            const INT64 * RESTRICT MM_CELLS,
                  REAL  * RESTRICT TREE,
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
                TREE
            );

            
            // add the new contrib
            kernel(
                NEW_P,
                &Q[IX],
                NEW_MM_CELLS,
                TREE
            );

 
            std::chrono::high_resolution_clock::time_point _loop_timer_t1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> _loop_timer_res = _loop_timer_t1 - _loop_timer_t0;
            *TIME_TAKEN = (double) _loop_timer_res.count();
               
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
            THREE_R=self.R*3,
            WIDTHS_X=widths_x,
            WIDTHS_Y=widths_y,
            WIDTHS_Z=widths_z,
            NCELLS_X=ncells_x,
            NCELLS_Y=ncells_y,
            NCELLS_Z=ncells_z,
            LEVEL_OFFSETS=level_offsets
        )

        self._indirect_accept_lib = lib.build.simple_lib_creator('#include <math.h>\n#include <chrono>', src)['indirect_accept']
