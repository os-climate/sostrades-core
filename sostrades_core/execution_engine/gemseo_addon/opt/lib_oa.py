'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/12-2023/11/02 Copyright 2023 Capgemini

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
from sostrades_core.execution_engine.gemseo_addon.opt.core.OuterApproximationSolver import OuterApproximationSolver
"""
Outer Approximation library
"""


from builtins import super, zip
import logging

from future import standard_library
from numpy import isfinite, real

from gemseo.algos.opt.opt_lib import OptimizationLibrary

standard_library.install_aliases()


LOGGER = logging.getLogger("gemseo.addons.opt.lib_oa")


class OuterApproximationOpt(OptimizationLibrary):
    """Outer Approximation optimization library interface.

    See OptimizationLibrary.
    """

    LIB_COMPUTE_GRAD = False

    OPTIONS_MAP = {OptimizationLibrary.MAX_ITER: "max_iter",
                   OptimizationLibrary.F_TOL_REL: "ftol_rel",
                   OptimizationLibrary.MAX_FUN_EVAL: "maxfun"
                   }

    def __init__(self):
        '''
        Constructor

        Generate the library dict, contains the list
        of algorithms with their characteristics:

        - does it require gradient
        - does it handle equality constraints
        - does it handle inequality constraints

        '''
        super(OuterApproximationOpt, self).__init__()
        self.lib_dict = {
            'OuterApproximation':
            {self.INTERNAL_NAME: "OuterApproximation",
             self.REQUIRE_GRAD: True,
             self.POSITIVE_CONSTRAINTS: False,
             self.HANDLE_EQ_CONS: False,
             self.HANDLE_INEQ_CONS: True,
             self.DESCRIPTION: 'Outer Approximation algorithm implementation',
             },

        }

    def _get_options(self, 
                     max_iter=999, 
                     ftol_rel=1e-10,  # pylint: disable=W0221
                     normalize_design_space=False,
                     **kwargs):
        r"""Sets the options default values

        To get the best and up to date information about algorithms options,
        go to scipy.optimize documentation:
        https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html

        :param max_iter: maximum number of iterations, ie unique calls to f(x)
        :type max_iter: int
        :param ftol_rel: stop criteria, relative tolerance on the
               objective function,
               if abs(f(xk)-f(xk+1))/abs(f(xk))<= ftol_rel: stop
               (Default value = 1e-9)
        :param max_fun_eval: internal stop criteria on the
               number of algorithm outer iterations (Default value = 999)
        :type max_fun_eval: int
        :param normalize_design_space: If True, scales variables in [0, 1]
        :type normalize_design_space: bool
        :param kwargs: other algorithms options
        :tupe kwargs: kwargs
        """
        popts = self._process_options(max_iter=max_iter, 
                                      ftol_rel=ftol_rel,
                                      normalize_design_space=normalize_design_space,
                                      ** kwargs)
        return popts

    def _run(self, **options):
        """Runs the algorithm,

        :param options: the options dict for the algorithm
        """
        normalize_ds = options.get(self.NORMALIZE_DESIGN_SPACE_OPTION, True)
        # Get the normalized bounds:
        x_0, l_b, u_b = self.get_x0_and_bounds_vects(normalize_ds)
        # Replace infinite values with None:
        l_b = [val if isfinite(val) else None for val in l_b]
        u_b = [val if isfinite(val) else None for val in u_b]
        bounds = list(zip(l_b, u_b))
        
        # execute the optimization
        solver = OuterApproximationSolver(self.problem)
        solver.set_options(**options)#**options
        solver.init_solver()
        solver.solve()

        return self.get_optimum_from_database()
