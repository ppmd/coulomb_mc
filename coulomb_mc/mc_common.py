

import numpy as np
from ppmd import opt




PROFILE = opt.PROFILE


class MCCommon:

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

    def _get_self_interaction(self, px, pos):

        charge = self.charges[px, 0]
        old_pos = self.positions[px, :]

        return charge * charge / np.linalg.norm(old_pos.ravel() - pos.ravel())


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


