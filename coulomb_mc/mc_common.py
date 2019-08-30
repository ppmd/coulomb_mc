

import numpy as np
from ppmd import opt, lib
from enum import Enum

from cgen import Module, Define, Include

import ctypes

REAL = ctypes.c_double
INT64 = ctypes.c_int64


PROFILE = opt.PROFILE

class BCType(Enum):
    """
    Enum to indicate boundary condition type.
    """

    PBC = 'pbc'
    """Fully periodic boundary conditions"""
    FREE_SPACE = 'free_space'
    """Free-space, e.g. vacuum, boundary conditions."""
    NEAREST = '27'
    """Primary image and the surrounding 26 nearest neighbours."""



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

        if self.boundary_condition == BCType.FREE_SPACE:

            return charge * charge / np.linalg.norm(old_pos.ravel() - pos.ravel())
        
        elif self.boundary_condition == BCType.PBC:
            
            old_position = np.array((old_pos[0], old_pos[1], old_pos[2]), REAL)
            new_position = np.array((pos[0], pos[1], pos[2]), REAL)

            return_energy = REAL(0)

            self._lr_si_lib(
                INT64(0),
                old_position.ctypes.get_as_parameter(),
                new_position.ctypes.get_as_parameter(),
                REAL(charge),
                REAL(self.lr_energy),
                self.mvector.ctypes.get_as_parameter(),
                self.evector.ctypes.get_as_parameter(),
                self.solver.lrc.linop_data.ctypes.get_as_parameter(),
                self.solver.lrc.linop_indptr.ctypes.get_as_parameter(),
                self.solver.lrc.linop_indices.ctypes.get_as_parameter(),
                ctypes.byref(return_energy)
            )
            

            return return_energy.value
        
        else:

            raise RuntimeError('Unknown boundary condition type.')




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


    def _init_lr_correction_libs(self):

        sph_gen = self.sph_gen

        def _re_lm(l, m): return l**2 + l + m

        assign_gen =  'double rhol = charge;\n'
        for lx in range(self.L):
            for mx in range(-lx, lx+1):

                res, ims = sph_gen.get_y_sym(lx, -mx)
                offset = _re_lm(lx, mx)

                assign_gen += ''.join(['MULTIPOLE[{}] += {} * rhol;\n'.format(*args) for args in (
                        (offset, str(res)),
                        (offset + self.L**2, str(ims))
                    )
                ])

                res, ims = sph_gen.get_y_sym(lx, mx)
                assign_gen += ''.join(['DOT_VEC[{}] += {} * rhol;\n'.format(*args) for args in (
                        (offset, str(res)),
                        (offset + self.L**2, '-1.0 * ' + str(ims))
                    )
                ])

            assign_gen += 'rhol *= radius;\n'


        co = (
            ( -1, -1, -1),
            (  0, -1, -1),
            (  1, -1, -1),
            ( -1,  0, -1),
            (  0,  0, -1),
            (  1,  0, -1),
            ( -1,  1, -1),
            (  0,  1, -1),
            (  1,  1, -1),

            ( -1, -1,  0),
            (  0, -1,  0),
            (  1, -1,  0),
            ( -1,  0,  0),
            (  0,  0,  0),
            (  1,  0,  0),
            ( -1,  1,  0),
            (  0,  1,  0),
            (  1,  1,  0),

            ( -1, -1,  1),
            (  0, -1,  1),
            (  1, -1,  1),
            ( -1,  0,  1),
            (  0,  0,  1),
            (  1,  0,  1),
            ( -1,  1,  1),
            (  0,  1,  1),
            (  1,  1,  1),
        )

        new27direct = 0.0
        ex = self.domain.extent
        for ox in co:
            # image of old pos
            dox = np.array((ex[0] * ox[0], ex[1] * ox[1], ex[2] * ox[2]))
            if ox != (0,0,0):
                new27direct -= 1.0 / np.linalg.norm(dox)

        offset_consts = ''
        bc27 = ''

        for oxi, ox in enumerate(co):

            offset_consts += '''
            const REAL dox{oxi} = EX * {OX};
            const REAL doy{oxi} = EY * {OY};
            const REAL doz{oxi} = EZ * {OZ};
            '''.format(
                oxi=str(oxi),
                OX=str(ox[0]),
                OY=str(ox[1]),
                OZ=str(ox[2]),
            )

            bc27 += '''
            const REAL dpx{oxi} = dox{oxi} + opx;
            const REAL dpy{oxi} = doy{oxi} + opy;
            const REAL dpz{oxi} = doz{oxi} + opz;

            const REAL ddx{oxi} = dpx{oxi} - npx;
            const REAL ddy{oxi} = dpy{oxi} - npy;
            const REAL ddz{oxi} = dpz{oxi} - npz;
            
            const REAL o_bbp{oxi} = 1.0 / sqrt(ddx{oxi}*ddx{oxi} + ddy{oxi}*ddy{oxi} + ddz{oxi}*ddz{oxi});
            energy27 += o_bbp{oxi};
            '''.format(
                oxi=str(oxi)
            )



        src = r'''


        static inline REAL apply_dipole_correction_split(
            const REAL * RESTRICT M,
            const REAL * RESTRICT E
        ){{
            
            REAL tmp = 0.0;

            tmp += (DIPOLE_SX * M[RE_1P1]) * E[RE_1P1];
            tmp += (DIPOLE_SX * M[RE_1P1]) * E[RE_1N1];
        
            tmp -= (DIPOLE_SY * M[IM_1P1]) * E[IM_1P1];
            tmp += (DIPOLE_SY * M[IM_1P1]) * E[IM_1N1];

            tmp += (DIPOLE_SZ * M[RE_1_0]) * E[RE_1_0];

            return tmp;
        }}

    
        static inline REAL linop_csr_both(
            const REAL * RESTRICT linop_data,
            const INT64 * RESTRICT linop_indptr,
            const INT64 * RESTRICT linop_indices,
            const REAL * RESTRICT x1,
            const REAL * RESTRICT E
        ){{
            
            INT64 data_ind = 0;
            REAL dot_tmp = 0.0;

            for(INT64 row=0 ; row<HALF_NCOMP ; row++){{

                REAL row_tmp_1 = 0.0;
                REAL row_tmp_2 = 0.0;

                for(INT64 col_ind=linop_indptr[row] ; col_ind<linop_indptr[row+1] ; col_ind++){{
                    const INT64 col = linop_indices[data_ind];
                    const REAL data = linop_data[data_ind];
                    data_ind++;
                    row_tmp_1 += data * x1[col];
                    row_tmp_2 += data * x1[col  + HALF_NCOMP];
                }}

                dot_tmp += row_tmp_1 * E[row] + row_tmp_2 * E[row + HALF_NCOMP];
            }}

            return dot_tmp;
        }}

        
        static inline void vector_diff(
            const REAL dx,
            const REAL dy,
            const REAL dz,
            const REAL charge,
                  REAL * MULTIPOLE,
                  REAL * DOT_VEC
        ){{
            const double xy2 = dx * dx + dy * dy;
            const double radius = sqrt(xy2 + dz * dz);
            const double theta = atan2(sqrt(xy2), dz);
            const double phi = atan2(dy, dx);
            {SPH_GEN}
            {ASSIGN_GEN}

        }}


        static inline REAL lr_energy_diff(
            const INT64             accept_flag,
            const REAL  * RESTRICT  old_position,
            const REAL  * RESTRICT  new_position,
            const REAL              charge,
            const REAL              old_energy,
                  REAL  * RESTRICT  existing_multipole,
                  REAL  * RESTRICT  existing_evector,
            const REAL  * RESTRICT  linop_data,
            const INT64 * RESTRICT  linop_indptr,
            const INT64 * RESTRICT  linop_indices
        ){{


            REAL mvector[NCOMP];
            REAL evector[NCOMP];
                
            // copy the existing vectors
            for(int ix=0 ; ix<NCOMP ; ix++){{
                mvector[ix] = existing_multipole[ix];
                evector[ix] = existing_evector[ix];
            }}


            // remove the old contribution
            vector_diff(old_position[0], old_position[1], old_position[2], -1.0 * charge, mvector, evector);
            // add the new contribution
            vector_diff(new_position[0], new_position[1], new_position[2], charge, mvector, evector);


            // cheap way to reuse this code for accepts
            if (accept_flag > 0){{
                
                for(int ix=0 ; ix<NCOMP ; ix++){{
                    existing_multipole[ix] = mvector[ix];
                    existing_evector[ix] = evector[ix];
                }}

                return 0.0;
            }}



            // apply the long range linear operator and get the new energy (minus dipole correction)
            REAL new_energy = 0.5 * linop_csr_both(
                linop_data, linop_indptr, linop_indices,
                mvector,
                evector
            );
            
            // add the dipole correction
            new_energy += 0.5 * apply_dipole_correction_split(
                mvector,
                evector
            );

            return new_energy - old_energy;

        }}
        
        
        
        static inline REAL self_contributon(
            const REAL  * RESTRICT  old_position,
            const REAL  * RESTRICT  new_position,
            const REAL              charge
        ){{
                
            const REAL opx = old_position[0];
            const REAL opy = old_position[1];
            const REAL opz = old_position[2];

            const REAL npx = new_position[0];
            const REAL npy = new_position[1];
            const REAL npz = new_position[2];

            {OFFSET_CONSTS}


            REAL energy27 = (DOMAIN_27_ENERGY);

            {BC27}


            return energy27 * charge * charge;

        }}


        extern "C" int lr_self_interaction(
            const INT64             accept_flag,
            const REAL  * RESTRICT  old_position,
            const REAL  * RESTRICT  new_position,
            const REAL              charge,
            const REAL              old_energy,
                  REAL  * RESTRICT  existing_multipole,
                  REAL  * RESTRICT  existing_evector,
            const REAL  * RESTRICT  linop_data,
            const INT64 * RESTRICT  linop_indptr,
            const INT64 * RESTRICT  linop_indices,
                  REAL  * RESTRICT  return_energy
        ){{

            REAL tmpu0 = lr_energy_diff(accept_flag, old_position, new_position, charge, old_energy, 
                existing_multipole, existing_evector, linop_data, linop_indptr, linop_indices);
            
            REAL tmpu1 = (accept_flag < 1) ? self_contributon(old_position, new_position, charge) : 0.0 ;

            printf("lr contrib %f, self interaction %f\n", tmpu0, tmpu1);
            
            *return_energy = -1.0 * tmpu0 + tmpu1;

            return 0;
        }}  



        '''.format(
            SPH_GEN=str(sph_gen.module),
            ASSIGN_GEN=str(assign_gen),
            OFFSET_CONSTS=str(offset_consts),
            BC27=str(bc27)
        )


        header = str(
            Module((
                Include('omp.h'),
                Include('stdio.h'),
                Include('math.h'),
                Define('INT64', 'int64_t'),
                Define('REAL', 'double'),
                Define('NCOMP', str(self.ncomp)),
                Define('HALF_NCOMP', str(self.L**2)),
                Define('DIPOLE_SX', str(self.lrc.dipole_correction[0])),
                Define('DIPOLE_SY', str(self.lrc.dipole_correction[1])),
                Define('DIPOLE_SZ', str(self.lrc.dipole_correction[2])),
                Define('RE_1P1', str(_re_lm(1, 1))),
                Define('RE_1_0', str(_re_lm(1, 0))),
                Define('RE_1N1', str(_re_lm(1,-1))),
                Define('IM_1P1', str(_re_lm(1, 1) + self.L**2)),
                Define('IM_1_0', str(_re_lm(1, 0) + self.L**2)),
                Define('DOMAIN_27_ENERGY', str(new27direct)),
                Define('IM_1N1', str(_re_lm(1,-1) + self.L**2)),
                Define('EX', self.domain.extent[0]),
                Define('EY', self.domain.extent[1]),
                Define('EZ', self.domain.extent[2]),
            ))
        )


        self._lr_si_lib = lib.build.simple_lib_creator(header_code=header, src_code=src)['lr_self_interaction']














