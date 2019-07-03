

import numpy as np
from ppmd import data, loop, kernel, access, lib
from ppmd.coulomb import fmm_interaction_lists, octal
from coulomb_kmc import kmc_expansion_tools, common
import ctypes
import math

from coulomb_mc import mc_expansion_tools

Constant = kernel.Constant
REAL = ctypes.c_double
INT64 = ctypes.c_int64

class MCFMM:

    def __init__(self, positions, charges, domain, boundary_condition, r, l):

        self.positions = positions
        self.charges = charges
        self.domain = domain
        self.boundary_condition = boundary_condition
        self.R = r
        self.L = l
        self.ncomp = (self.L ** 2) * 2

        self.subdivision = (2, 2, 2)
        
        # tree
        self.tree = octal.OctalTree(self.R, domain.comm)
        self.tree_local = octal.OctalDataTree(self.tree, self.ncomp, 'plain', REAL)
        
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
        g._mc_nexp = pd(ncomp=1, dtype=INT64)
        g._mc_charge = pd(ncomp=l, dtype=REAL)
        g._mc_radius = pd(ncomp=l, dtype=REAL)
        g._mc_theta = pd(ncomp=l, dtype=REAL)
        g._mc_phi = pd(ncomp=l, dtype=REAL)
        g._mc_level = pd(ncomp=l, dtype=INT64)
        g._mc_cx = pd(ncomp=l, dtype=INT64)
        g._mc_cy = pd(ncomp=l, dtype=INT64)
        g._mc_cz = pd(ncomp=l, dtype=INT64)
        g._mc_cl = pd(ncomp=l, dtype=INT64)
        g._mc_ptrs = pd(ncomp=ptr_l, dtype=INT64)

        self.energy = None

        self.direct_map = {}

        self._cell_bin_loop = None
        self._cell_bin_format_lib = None
        self._init_bin_loop()


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
        return [
            int(cx) // (2**(self.R - level - 1)) for cx in cell
        ]


    def _get_child_index(self, cell):
        t = [int(cx) % 2 for cx in cell]
        return t[0] + 2 * (t[1] + 2 * t[2])


    def _setup_tree(self):

        g = self.positions.group
        N = self.positions.npart_local

        start_ptrs = np.zeros(self.R, ctypes.c_void_p)
        for rx in range(self.R):
            start_ptrs[rx] = self.tree_local[rx].ctypes.get_as_parameter().value

        self._cell_bin_loop.execute()

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
 
        # indirect part ( and bin into cells)
        for px in range(N):
            pos = self.positions[px,:].copy()
            charge = float(self.charges[px, 0])

            # cell on finest level
            cell = tuple(C[px, :])

            if cell in self.direct_map.keys():
                self.direct_map[cell].append(px)
            else:
                self.direct_map[cell] = [px]


        # indirect part 
        for px in range(N):
            energy += self._get_old_energy(px)


        return 0.5 * energy
    

    def _get_old_energy(self, px):

        tmp = np.zeros(self.ncomp, REAL)
        N = self.positions.npart_local
        g = self.positions.group
        C = g._mc_fmm_cells
        
        sl = 2**(self.R - 1)
        energy = 0.0

        pos = self.positions[px,:].copy()

        charge = float(self.charges[px, 0])

        # cell on finest level
        cell = (C[px, 0], C[px, 1], C[px, 2])

        for level in range(self.R):
            
            cell_level = self._get_parent(cell, level)
            sph = self._get_cell_disp(cell_level, pos, level)
            tmp[:] = 0.0
            self.lee.dot_vec(sph, charge, tmp)

            energy += np.dot(tmp.ravel(), self.tree_local[level][cell_level[2], cell_level[1], cell_level[0], :])

        for ox in self.il[1]:

            ccc = (
                cell[0] + ox[0],
                cell[1] + ox[1],
                cell[2] + ox[2],
            )
            if ccc[0] < 0: continue
            if ccc[1] < 0: continue
            if ccc[2] < 0: continue
            if ccc[0] >= sl: continue
            if ccc[1] >= sl: continue
            if ccc[2] >= sl: continue

            ccc = tuple(ccc)
            if ccc not in self.direct_map: continue

            for jx in self.direct_map[ccc]:
                if jx == px: continue

                energy += charge * self.charges[jx, 0] / np.linalg.norm(pos - self.positions[jx, :])

        return energy


    def _get_new_energy(self, px, pos):

        tmp = np.zeros(self.ncomp, REAL)
        N = self.positions.npart_local
        g = self.positions.group
        
        sl = 2**(self.R - 1)
        energy = 0.0

        charge = float(self.charges[px, 0])

        # cell on finest level
        cell = self._get_cell(pos)

        for level in range(self.R):
            
            cell_level = self._get_parent(cell, level)
            sph = self._get_cell_disp(cell_level, pos, level)
            tmp[:] = 0.0
            self.lee.dot_vec(sph, charge, tmp)

            energy += np.dot(tmp.ravel(), self.tree_local[level][cell_level[2], cell_level[1], cell_level[0], :])

        for ox in self.il[1]:

            ccc = (
                cell[0] + ox[0],
                cell[1] + ox[1],
                cell[2] + ox[2],
            )
            if ccc[0] < 0: continue
            if ccc[1] < 0: continue
            if ccc[2] < 0: continue
            if ccc[0] >= sl: continue
            if ccc[1] >= sl: continue
            if ccc[2] >= sl: continue

            ccc = tuple(ccc)
            if ccc not in self.direct_map: continue

            for jx in self.direct_map[ccc]:
                energy += charge * self.charges[jx, 0] / np.linalg.norm(pos - self.positions[jx, :])

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

        return  new_energy - old_energy - self_energy


    def accept(self, move, energy_diff=None):
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
        if tnew in self.direct_map.keys():
            self.direct_map[tnew].append(px)
        else:
            self.direct_map[tnew] = [px]
     

        # to pass into lib
        charge = np.zeros(self.il_max_len * self.R, REAL)
        radius = np.zeros_like(charge)
        theta = np.zeros_like(charge)
        phi = np.zeros_like(charge)
        ptrs = np.zeros(self.il_max_len * self.R, ctypes.c_void_p)

        n = 0
        # remove old contrib
        for level in range(self.R):
            
            # cell on level 
            cell_level = self._get_parent(old_cell, level)
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

                sph = self._get_cell_disp(ccc, old_pos, level)
                #self.lee.local_exp(sph, -q, self.tree_local[level][ccc[2], ccc[1], ccc[0], :])

                charge[n] = -q
                radius[n] = sph[0]
                theta[n] = sph[1]
                phi[n] = sph[2]
                ptrs[n] = self.tree_local[level][ccc[2], ccc[1], ccc[0], :].ctypes.get_as_parameter().value
                n += 1

        self.mc_lee.compute_local_exp(n, charge, radius, theta, phi, ptrs)

        n = 0
        # add new contrib
        for level in range(self.R):
            
            # cell on level 
            cell_level = self._get_parent(new_cell, level)
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

                sph = self._get_cell_disp(ccc, new_pos, level)
                #self.lee.local_exp(sph, q, self.tree_local[level][ccc[2], ccc[1], ccc[0], :])

                charge[n] = q
                radius[n] = sph[0]
                theta[n] = sph[1]
                phi[n] = sph[2]
                ptrs[n] = self.tree_local[level][ccc[2], ccc[1], ccc[0], :].ctypes.get_as_parameter().value
                n += 1

        self.mc_lee.compute_local_exp(n, charge, radius, theta, phi, ptrs)




        with g._mc_fmm_cells.modify_view() as m:
            m[px, :] = new_cell

        
        # move the particle in the dats
        with self.positions.modify_view() as m:
            m[px, :] = new_pos.copy()














