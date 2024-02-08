'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/12-2023/11/03 Copyright 2023 Capgemini

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
# -*-mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8 -*-
import logging
import numpy as np
from copy import copy
from sostrades_core.execution_engine.parallel_execution.sos_parallel_execution import SoSDiscParallelExecution
from sostrades_core.tools.conversion.conversion_sostrades_sosgemseo import convert_array_into_new_type

"""
A chain of MDAs to build hybrids of MDA algorithms sequentially
***************************************************************
"""

from gemseo.core.discipline import MDODiscipline
from gemseo.mda.newton import MDARoot

LOGGER = logging.getLogger("gemseo.addons.mda.pure_newton_raphson")


class PureNewtonRaphson(MDARoot):
    """
    Pure NewtonRaphson solver based on Taylor's theorem.
    """

    def __init__(
        self,
        disciplines,  # type: Sequence[MDODiscipline]
        max_mda_iter=10,  # type: int
        relax_factor=0.99,  # type: float
        name=None,  # type: Optional[str]
        grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,  # type: str
        linear_solver="DEFAULT",  # type: str
        tolerance=1e-6,  # type: float
        linear_solver_tolerance=1e-12,  # type: float
        warm_start=False,  # type: bool
        use_lu_fact=False,  # type: bool
        coupling_structure=None,  # type: Optional[MDOCouplingStructure]
        log_convergence=False,  # type:bool
        linear_solver_options=None,  # type: Mapping[str,Any]
            n_processes=1
    ):
        """
        Args:
            relax_factor: The relaxation factor in the Newton step.
        """

        self.n_processes = n_processes

        super(PureNewtonRaphson, self).__init__(
            disciplines,
            max_mda_iter=max_mda_iter,
            name=name,
            grammar_type=grammar_type,
            tolerance=tolerance,
            linear_solver_tolerance=linear_solver_tolerance,
            warm_start=warm_start,
            use_lu_fact=use_lu_fact,
            linear_solver=linear_solver,
            linear_solver_options=linear_solver_options,
            coupling_structure=coupling_structure,
            log_convergence=log_convergence,
        )
        self.relax_factor = self.__check_relax_factor(relax_factor)
        self.linear_solver = linear_solver

        # break the object link before update the dict object
        self.linear_solver_options = copy(self.linear_solver_options)
        self.linear_solver_options.update(
            {'tol': self.linear_solver_tolerance})

        self.parallel_execution = SoSDiscParallelExecution(
            disciplines, n_processes=self.n_processes, use_threading=True
        )

        self.assembly.parallel_linearize.configure_linearize_options(
            force_no_exec=True)

    @staticmethod
    def __check_relax_factor(
        relax_factor,  # type: float
    ):  # type:(...) -> float
        """Check that the relaxation factor in the Newton step is in (0, 1].

        Args:
            relax_factor: The relaxation factor.
        """
        if relax_factor <= 0.0 or relax_factor > 1:
            raise ValueError(
                "Newton relaxation factor should belong to (0, 1] "
                "(current value: {}).".format(relax_factor)
            )
        return relax_factor

    def _run(self):  # type: (...) -> None
        '''
            R = self.__R(self.__W)
            if self.__dRdW is None:
                dRdW = FD_grad.grad_f(self.__W)

            step = self.get_relax_factor() * -solve(dRdW, R)

            self.__W = self.__W + step

            if self.__Res0 is None:
                self.__Res0 = norm(R)
            # Compute stop criteria
            self.__residual = norm(R) / self.__Res0
            self.__residual_hist.append(self.__residual)
        '''
        # store initial residual
        current_iter = 1
        self.reset_disciplines_statuses()
        
        # build current_couplings: concatenated strong couplings, converted into arrays
        current_couplings, old_x_array = self._current_strong_couplings(return_converted_dict=True, update_dm=True)
        #self.execute_all_disciplines(self.local_data)
        
        while not self._termination(current_iter):

            # Set coupling variables as differentiated variables for gradient
            # computation
            self.assembly._add_differentiated_inouts(
                self.strong_couplings, self.strong_couplings, self.strong_couplings)

            # Compute all discipline gradients df(x)/dx with x
            self.assembly.linearize_all_disciplines(self.local_data, force_no_exec=True)

            # compute coupling_variables(x+k) for the residuals
            self.execute_all_disciplines(self.local_data)

            # build new_couplings after execution: concatenated strong couplings, converted into arrays
            new_couplings = self._current_strong_couplings(update_dm=True)

            # res = coupling_variables(x+k) - coupling_variables(x)
            res = new_couplings - current_couplings

            # compute_normed_residual
            self._compute_residual(
                current_couplings,
                new_couplings,
                current_iter,
                first=True,
                log_normed_residual=self.log_convergence,)

            if self._termination(current_iter):
                print(current_iter, self.normed_residual, self.tolerance)
                break
            # compute newton step with res and gradients computed with x=
            # coupling_variables(n)
            newton_step_dict = self.assembly.compute_newton_step_pure(
                res,
                self.strong_couplings,
                self.relax_factor,
                self.linear_solver,
                matrix_type=self.matrix_type,
                **self.linear_solver_options)

            # ynew = yk+1 + step
            # update current solution with Newton step
            # we update coupling_variables(n) with newton step
            for c_var, c_step in newton_step_dict.items():
                old_x_array[c_var] += c_step.real  # SoSTrades fix (.real)

            # convert old_x_array into SoSTrades types and store it into local_data for next execution
            self.local_data.update(convert_array_into_new_type(
                old_x_array, self.disciplines[0].reduced_dm))

            # store current_couplings for residual computation of next iteration
            current_couplings = np.hstack(list(old_x_array.values()))
            current_iter += 1
