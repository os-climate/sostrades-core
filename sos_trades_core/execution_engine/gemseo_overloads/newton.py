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
from gemseo.mda.newton import MDANewtonRaphson
from copy import deepcopy

def _newton_step(cls):  # type: (...) -> None
    """Execute the full Newton step.

    Compute the increment :math:`-[dR/dW]^{-1}.R` and run the disciplines.
    """
    # SoSTrades fix: pass linear solver tolerance to linear_solver_options...
    tol_dict = {'tol':cls.linear_solver_tolerance}
    cls.linear_solver_options.update(tol_dict)
    #
    newton_dict = cls.assembly.compute_newton_step(
        cls.local_data,
        cls.strong_couplings,
        cls.relax_factor,
        cls.linear_solver,
        matrix_type=cls.matrix_type,
        **cls.linear_solver_options
    )
    # update current solution with Newton step
    exec_data = deepcopy(cls.local_data)
    for c_var, c_step in newton_dict.items():
        exec_data[c_var] += c_step.real # SoSTrades fix (.real)
    cls.reset_disciplines_statuses()
    cls.execute_all_disciplines(exec_data)
    
# Set functions to the MDA Class
setattr(MDANewtonRaphson, "_newton_step", _newton_step)