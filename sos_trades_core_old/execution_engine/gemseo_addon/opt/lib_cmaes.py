'''
Copyright 2022 Airbus SAS

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
"""
scipy.optimize optimization library wrapper
"""


from builtins import super, zip
import logging

from future import standard_library
from numpy import isfinite, real

from gemseo.algos.opt.opt_lib import OptimizationLibrary

import cma

standard_library.install_aliases()


LOGGER = logging.getLogger("GEMS")


class CMAESOpt(OptimizationLibrary):
    """Scipy optimization library interface.

    See OptimizationLibrary.
    """

    LIB_COMPUTE_GRAD = True

    OPTIONS_MAP = {OptimizationLibrary.MAX_ITER: "maxiter",
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
        super(CMAESOpt, self).__init__()
        doc = 'https://docs.scipy.org/doc/scipy/reference/'
        self.lib_dict = {
            'CMAES':
            {self.INTERNAL_NAME: "CMAES",
             self.REQUIRE_GRAD: False,
             self.POSITIVE_CONSTRAINTS: True,
             self.HANDLE_EQ_CONS: False,
             self.HANDLE_INEQ_CONS: False,
             self.DESCRIPTION: 'Sequential Least-Squares Quadratic '
             'Programming (SLSQP) implemented in '
             'the SciPy library',
             self.WEBSITE: doc + 'optimize.minimize-slsqp.html',
             },

        }

    def _get_options(self, max_iter=999, ftol_rel=1e-10,  # pylint: disable=W0221
                     max_fun_eval=999, sigma=0.1, normalize_design_space=False, population_size=20,
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
        popts = self._process_options(max_iter=max_iter, ftol_rel=ftol_rel,
                                      max_fun_eval=max_fun_eval,
                                      sigma=sigma, normalize_design_space=normalize_design_space,
                                      population_size=population_size,
                                      ** kwargs)
        return popts

    def _run(self, **options):
        """Runs the algorithm, to be overloaded by subclasses

        :param options: the options dict for the algorithm
        """
        # remove normalization from options for algo
        normalize_ds = options.pop(self.NORMALIZE_DESIGN_SPACE_OPTION, True)
        # Get the normalized bounds:
        x_0, l_b, u_b = self.get_x0_and_bounds_vects(normalize_ds=normalize_ds)

        def real_part_fun(x_vect):
            """
            Wraps the function and returns the real part
            """
            return real(self.problem.objective.func(x_vect))

        fun = real_part_fun
        # GEMSEO is in charge of ensuring max iterations, since it may
        # have a different definition of iterations, such as for SLSQP
        # for instance which counts duplicate calls to x as a new iteration
        options[self.OPTIONS_MAP[self.MAX_ITER]] = 10000
        sigma = options['sigma']
        ftol = options['ftol_rel']
        popsize = options['population_size']
        es = cma.CMAEvolutionStrategy(
            x_0, sigma, {'tolfun': ftol, 'bounds': [l_b, u_b], 'popsize': popsize})
        es.optimize(fun)

        return self.get_optimum_from_database(es.result
                                              )
