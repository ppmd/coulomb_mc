__author__ = "W.R.Saunders"


import ctypes
import numpy as np
from math import *

REAL = ctypes.c_double
INT64 = ctypes.c_int64

# cuda imports if possible
from ppmd.coulomb.sph_harm import *
from ppmd.lib.build import simple_lib_creator
from coulomb_kmc import common


class LocalExp:
    """
    Generates C code to manipulate and evaluate Local expansions.

    :arg int L: Number of expansion terms.
    """
    
    def __init__(self, L):
        self.L = L
        self._hmatrix_py = np.zeros((2*self.L, 2*self.L))
        def Hfoo(nx, mx): return sqrt(float(factorial(nx - abs(mx)))/factorial(nx + abs(mx)))
        for nx in range(self.L):
            for mx in range(-nx, nx+1):
                self._hmatrix_py[nx, mx] = Hfoo(nx, mx)
        
        self.sph_gen = SphGen(L-1)
        self._generate_host_libs()

    def _generate_host_libs(self):

        sph_gen = self.sph_gen

        def cube_ind(L, M):
            return ((L) * ( (L) + 1 ) + (M) )


        # --- lib to evaluate local expansions --- 
        assign_gen = ''
        for lx in range(self.L):
            for mx in range(-lx, lx+1):
                reL = SphSymbol('moments[{ind}]'.format(ind=cube_ind(lx, mx)))
                imL = SphSymbol('moments[IM_OFFSET + {ind}]'.format(ind=cube_ind(lx, mx)))
                reY, imY = sph_gen.get_y_sym(lx, mx)
                phi_sym = cmplx_mul(reL, imL, reY, imY)[0]
                assign_gen += 'tmp_energy += rhol * ({phi_sym});\n'.format(phi_sym=str(phi_sym))

            assign_gen += 'rhol *= radius;\n'

        src = """
        #define IM_OFFSET ({IM_OFFSET})
        #define REAL double
        #define INT64 int64_t

        {DECLARE} int local_eval(
            const INT64 n,
            const REAL * RESTRICT hradius,
            const REAL * RESTRICT htheta,
            const REAL * RESTRICT hphi,
            const REAL * RESTRICT * RESTRICT hmoments,
            REAL * RESTRICT out
        ){{

            
            #pragma omp parallel for
            for(INT64 ix=0 ; ix<n ; ix++){{
                const REAL radius = hradius[ix];
                const REAL theta = htheta[ix];
                const REAL phi = hphi[ix];
                const REAL * RESTRICT moments = hmoments[ix];

                {SPH_GEN}
                REAL rhol = 1.0;
                REAL tmp_energy = 0.0;
                {ASSIGN_GEN}
                out[ix] = tmp_energy;
            }}
            return 0;
        }}
        """

        src_internal = src.format(
            SPH_GEN=str(sph_gen.module),
            ASSIGN_GEN=str(assign_gen),
            IM_OFFSET=(self.L**2),
            DECLARE=r'extern "C"'
        )
        src_lib = src.format(
            SPH_GEN=str(sph_gen.module),
            ASSIGN_GEN=str(assign_gen),
            IM_OFFSET=(self.L**2),
            DECLARE=r'static inline'
        )

        header = str(sph_gen.header)


        self.create_local_eval_header = header
        self.create_local_eval_src = src_lib

        self._local_eval_lib = simple_lib_creator(header_code=header, src_code=src_internal)['local_eval']

        # lib to create local expansions
        

        assign_gen = 'const REAL iradius = 1.0/radius;\n'
        assign_gen += 'REAL rhol = iradius;\n'
        for lx in range(self.L):
            for mx in range(-lx, lx+1):
                assign_gen += 'out[{ind}] += {ylmm} * rhol * charge;\n'.format(
                        ind=cube_ind(lx, mx),
                        ylmm=str(sph_gen.get_y_sym(lx, -mx)[0])
                    )
                assign_gen += 'out[IM_OFFSET + {ind}] += {ylmm} * rhol * charge;\n'.format(
                        ind=cube_ind(lx, mx),
                        ylmm=str(sph_gen.get_y_sym(lx, -mx)[1])
                    )
            assign_gen += 'rhol *= iradius;\n'
        

        src = """
        #define IM_OFFSET ({IM_OFFSET})
        #define REAL double
        #define INT64 int64_t

        extern "C" int create_local_exp(
            const INT64 n,
            const REAL * RESTRICT hcharge,
            const REAL * RESTRICT hradius,
            const REAL * RESTRICT htheta,
            const REAL * RESTRICT hphi,
            REAL * RESTRICT * RESTRICT hout
        ){{

            #pragma omp parallel for
            for(INT64 ix=0 ; ix<n ; ix++){{
                const INT64 OFFSET = ix * IM_OFFSET * 2;
                const REAL charge = hcharge[ix];
                const REAL radius = hradius[ix];
                const REAL theta = htheta[ix];
                const REAL phi = hphi[ix];
                REAL * RESTRICT out = hout[ix];
                {SPH_GEN}
                {ASSIGN_GEN}
            }}
            return 0;
        }}
        """.format(
            SPH_GEN=str(sph_gen.module),
            ASSIGN_GEN=str(assign_gen),
            IM_OFFSET=(self.L**2),
        )
        header = str(sph_gen.header)


        self.create_local_exp_header = header
        self.create_local_exp_src = """
        #define IM_OFFSET ({IM_OFFSET})

        static inline void inline_local_exp(
            const double charge,
            const double radius,
            const double theta,
            const double phi,
            double * RESTRICT out
        ){{
            {SPH_GEN}
            {ASSIGN_GEN}
            return;
        }}
        """.format(
            SPH_GEN=str(sph_gen.module),
            ASSIGN_GEN=str(assign_gen),
            IM_OFFSET=(self.L**2),
        )


        self._local_create_lib = simple_lib_creator(header_code=header, src_code=src)['create_local_exp']
        
        # --- lib to evaluate a single local expansion --- 

        assign_gen = ''
        for lx in range(self.L):
            for mx in range(-lx, lx+1):
                reL = SphSymbol('moments[{ind}]'.format(ind=cube_ind(lx, mx)))
                imL = SphSymbol('moments[IM_OFFSET + {ind}]'.format(ind=cube_ind(lx, mx)))
                reY, imY = sph_gen.get_y_sym(lx, mx)
                phi_sym = cmplx_mul(reL, imL, reY, imY)[0]
                assign_gen += 'tmp_energy += rhol * ({phi_sym});\n'.format(phi_sym=str(phi_sym))

            assign_gen += 'rhol *= radius;\n'

        src = """
        #define IM_OFFSET ({IM_OFFSET})

        {DECLARE} int local_eval(
            const double radius,
            const double theta,
            const double phi,
            const double * RESTRICT moments,
            double * RESTRICT out
        ){{
            {SPH_GEN}
            double rhol = 1.0;
            double tmp_energy = 0.0;
            {ASSIGN_GEN}

            out[0] = tmp_energy;
            return 0;
        }}
        """
        
        src_lib = src.format(
            SPH_GEN=str(sph_gen.module),
            ASSIGN_GEN=str(assign_gen),
            IM_OFFSET=(self.L**2),
            DECLARE=r'static inline'
        )

        src = src.format(
            SPH_GEN=str(sph_gen.module),
            ASSIGN_GEN=str(assign_gen),
            IM_OFFSET=(self.L**2),
            DECLARE=r'extern "C"'
        )
        header = str(sph_gen.header)


        self.create_single_local_eval_header = header
        self.create_single_local_eval_src = src_lib

        self._single_local_eval_lib = simple_lib_creator(header_code=header, src_code=src)['local_eval']



    def compute_phi_local(self, n, radius, theta, phi, moments, out):
        """
        Evaluate local expansions at points.

        :arg n: Number of points,
        :arg radius: np.array n x 1, c_double of radii.
        :arg theta:  np.array n x 1, c_double of theta values.
        :arg phi:    np.array n x 1, c_double of phi values.
        :arg moments: np.array n x 1, c_void_p of pointers to LxLx2 c_double values.
        :arg out: np.array n x 1, c_double of output field evaluations.
        """

        self._local_eval_lib(
            INT64(n),
            radius.ctypes.get_as_parameter(),
            theta.ctypes.get_as_parameter(),
            phi.ctypes.get_as_parameter(),
            moments.ctypes.get_as_parameter(),
            out.ctypes.get_as_parameter()
        )



    def compute_local_exp(self, n, charge, radius, theta, phi, out):
        """
        Compute the local expansion that describes the field induced by a charge at the origin.

        :arg n: Number of charges.
        :arg charge:  np.array n x 1, c_double charge values.
        :arg radius:  np.array n x 1, c_double radii.
        :arg theta:   np.array n x 1, c_double theta values.
        :arg phi:     np.array n x 1, c_double phi values.
        :arg out:     np.array n x 1, c_void_p pointer to LxLx2 c_double values (MUST NOT ALIAS). I.E. pointers must be unique and not overlap.
        """

        self._local_create_lib(
            INT64(n),
            charge.ctypes.get_as_parameter(),
            radius.ctypes.get_as_parameter(),
            theta.ctypes.get_as_parameter(),
            phi.ctypes.get_as_parameter(),
            out.ctypes.get_as_parameter()
        )


