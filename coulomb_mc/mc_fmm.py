

import numpy as np
from ppmd import data
from ppmd.coulomb import fmm_interaction_lists, octal
from coulomb_kmc import kmc_expansion_tools, common
import ctypes

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
        
        # tree
        self.tree = octal.OctalTree(self.R, domain.comm)
        self.tree_local = octal.OctalDataTree(self.tree, self.ncomp, 'plain', REAL)
        
        # interaction lists
        self.il = fmm_interaction_lists.compute_interaction_lists(domain.extent)
        
        # expansion tools
        self.lee = kmc_expansion_tools.LocalExpEval(self.L)


        self.energy = None

        self.direct_map = {}
    

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
        g.mc_fmm_cells = data.ParticleDat(ncomp=3, dtype=INT64)

        
        N = self.positions.npart_local

        with g.mc_fmm_cells.modify_view() as gm:

            for px in range(N):
                pos = self.positions[px,:].copy()
                charge = float(self.charges[px, 0])

                # cell on finest level
                cell = self._get_cell(pos)
                gm[px, :] = cell

                for level in range(self.R):
                    
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
                        self.lee.local_exp(sph, charge, self.tree_local[level][ccc[2], ccc[1], ccc[0], :])



    
    def _compute_energy(self):

        N = self.positions.npart_local
        g = self.positions.group
        C = g.mc_fmm_cells

        tmp = np.zeros(self.ncomp, REAL)

        energy = 0.0
        

        # indirect part ( and bin into cells)
        for px in range(N):
            pos = self.positions[px,:].copy()
            charge = float(self.charges[px, 0])

            # cell on finest level
            cell = tuple(C[px, :])

            for level in range(self.R):
                
                cell_level = self._get_parent(cell, level)
                sph = self._get_cell_disp(cell_level, pos, level)
                tmp[:] = 0.0
                self.lee.dot_vec(sph, charge, tmp)

                energy += np.dot(tmp.ravel(), self.tree_local[level][cell_level[2], cell_level[1], cell_level[0], :])


            if cell in self.direct_map.keys():
                self.direct_map[cell].append(px)
            else:
                self.direct_map[cell] = [px]


        sl = 2**(self.R - 1)
        for px in range(N):
            pos = self.positions[px,:].copy()
            charge = float(self.charges[px, 0])

            # cell on finest level
            cell = C[px, :]            
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

        return 0.5 * energy



    def initialise(self):
        self._setup_tree()
        self.energy = self._compute_energy()


















