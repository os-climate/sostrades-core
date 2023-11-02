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
from scipy.optimize import fmin_tnc
"""
Projected Gradient library
"""


from builtins import super, zip
import logging

from future import standard_library
from numpy import isfinite, real

from gemseo.algos.opt.opt_lib import OptimizationLibrary

standard_library.install_aliases()


LOGGER = logging.getLogger("gemseo.addons.opt.lib_projected_gradient")


class ProjectedGradientOpt(OptimizationLibrary):
    """Projected Gradient optimization library interface.

    See OptimizationLibrary.
    """

    LIB_COMPUTE_GRAD = True

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
        super(ProjectedGradientOpt, self).__init__()
        doc = 'https://docs.scipy.org/doc/scipy/reference/'
        self.lib_dict = {
            'ProjectedGradient':
            {self.INTERNAL_NAME: "ProjectedGradient",
             self.REQUIRE_GRAD: True,
             self.POSITIVE_CONSTRAINTS: False,
             self.HANDLE_EQ_CONS: True,
             self.HANDLE_INEQ_CONS: True,
             self.DESCRIPTION: 'Projected conjugated Gradient algorithm implementation',
             self.WEBSITE: doc + 'generated/scipy.optimize.fmin_tnc.html',
             },

        }

    def _get_options(self, 
                     maxfun=999,
                     ftol=1e-10,  # pylint: disable=W0221
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
        :param kwargs: other algorithms options
        :tupe kwargs: kwargs
        """
        popts = self._process_options(maxfun=kwargs['max_iter'],
                                      ftol=ftol,
                                      ** kwargs)
        return popts

    def _process_options(
        self, **options  # type:Any
    ):  # type: (...) -> Dict[str, Any]
        options = OptimizationLibrary._process_options(self, **options)
        self.OPTIONS_MAP[self.MAX_ITER] = options['maxfun']
        self.OPTIONS_MAP[self.NORMALIZE_DESIGN_SPACE_OPTION] = True
        options.update(self.OPTIONS_MAP)
        return options

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

        options.pop(self.NORMALIZE_DESIGN_SPACE_OPTION)
        maxfun = options[self.MAX_ITER]
        options.pop(self.MAX_ITER)
        options.pop("maxfun")

        options.pop("max_fun_eval")
        # execute the optimization
        self._ftol_rel = 1e-10
        options.pop("ftol_rel")

        x_star, nfev, status = fmin_tnc(func=self.problem.objective.func,
                                        x0=x_0,
                                        fprime=self.problem.objective.jac,
                                        maxfun=maxfun,
                                        bounds=bounds,
                                        **options)

        x_opt = self.problem.design_space.project_into_bounds(x_star)
        val_opt, jac_opt = self.problem.evaluate_functions(
            x_vect=x_opt,
            eval_jac=True,
            eval_obj=True,
            normalize=False,
            no_db_no_norm=True,
        )
        f_opt = val_opt[self.problem.objective.outvars[0]]
        constraints_values = {
            key: val_opt[key] for key in self.problem.get_constraints_names()
        }
        constraints_grad = {
            key: jac_opt[key] for key in self.problem.get_constraints_names()
        }
        is_feasible = self.problem.is_point_feasible(val_opt)
        from gemseo.algos.opt_result import OptimizationResult
        optim_result = OptimizationResult(
            x_0=x_0,
            x_opt=x_opt,
            f_opt=f_opt,
            status=status,
            constraints_values=constraints_values,
            constraints_grad=constraints_grad,
            optimizer_name=self.algo_name,
            message="",
            n_obj_call=None,
            n_grad_call=None,
            n_constr_call=None,
            is_feasible=is_feasible,
        )
        return optim_result
