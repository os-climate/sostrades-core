'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/12-2024/05/16 Copyright 2023 Capgemini

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

from __future__ import annotations

import logging
from copy import copy
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
from gemseo.mda.base_mda_root import BaseMDARoot

from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.tools.conversion.conversion_sostrades_sosgemseo import (
    convert_array_into_new_type,
    convert_new_type_into_array,
)

if TYPE_CHECKING:
    from gemseo.core.coupling_structure import CouplingStructure
    from gemseo.core.discipline import MDODiscipline

LOGGER = logging.getLogger("gemseo.addons.mda.pure_newton_raphson")


class PureNewtonRaphson(BaseMDARoot):
    """Pure NewtonRaphson solver based on Taylor's theorem."""

    def __init__(
        self,
        disciplines: Sequence[MDODiscipline],  # type: Sequence[MDODiscipline]
        max_mda_iter: int = 10,  # type: int
        over_relaxation_factor: float = 0.99,  # type: float
        name: str | None = None,  # type: Optional[str]
        grammar_type: str = ProxyDiscipline.SOS_GRAMMAR_TYPE,  # type: str
        linear_solver: str = "DEFAULT",  # type: str
        tolerance: float = 1e-6,  # type: float
        linear_solver_tolerance: float = 1e-12,  # type: float
        scaling_method: BaseMDARoot.ResidualScaling = BaseMDARoot.ResidualScaling.N_COUPLING_VARIABLES,
        warm_start: bool = False,  # type: bool
        use_lu_fact: bool = False,  # type: bool
        coupling_structure: CouplingStructure | None = None,  # type: Optional[MDOCouplingStructure]
        log_convergence: bool = False,  # type:bool
        linear_solver_options: Mapping[str, Any] | None = None,  # type: Mapping[str,Any]
        n_processes=1,
    ) -> None:
        """
        Args:
            relax_factor: The relaxation factor in the Newton step.
        """
        self.n_processes = n_processes

        super().__init__(
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
        self.over_relaxation_factor = self.__check_relax_factor(over_relaxation_factor)
        self.linear_solver = linear_solver
        self.scaling = scaling_method

        # break the object link before update the dict object
        self.linear_solver_options = copy(self.linear_solver_options)
        self.linear_solver_options.update({'tol': self.linear_solver_tolerance})

        # self.parallel_execution = SoSDiscParallelExecution(
        #     disciplines, n_processes=self.n_processes, use_threading=True
        # )

        # self.assembly.parallel_linearize.configure_linearize_options(
        #     force_no_exec=True)

    @staticmethod
    def __check_relax_factor(
        relax_factor,  # type: float
    ):  # type:(...) -> float
        """Check that the relaxation factor in the Newton step is in (0, 1].

        Args:
            relax_factor: The relaxation factor.
        """
        if relax_factor <= 0.0 or relax_factor > 1:
            msg = f"Newton relaxation factor should belong to (0, 1] (current value: {relax_factor})."
            raise ValueError(msg)
        return relax_factor

    def _run(self):  # type: (...) -> None
        """
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
        """
        # store initial residual
        current_iter = 1
        self.reset_disciplines_statuses()

        # build current_couplings: concatenated strong couplings, converted into arrays
        current_couplings, old_x_array = self._current_strong_couplings(return_converted_dict=True, update_dm=True)
        # self.execute_all_disciplines(self.local_data)

        while not self._termination(current_iter):
            # Set coupling variables as differentiated variables for gradient
            # computation
            self.assembly._add_differentiated_inouts(
                self.strong_couplings, self.strong_couplings, self.strong_couplings
            )

            # Compute all discipline gradients df(x)/dx with x
            self.linearize_all_disciplines(self.local_data, execute=False)

            # compute coupling_variables(x+k) for the residuals
            self.execute_all_disciplines(self.local_data)

            # build new_couplings after execution: concatenated strong couplings, converted into arrays
            new_couplings = self._current_strong_couplings()

            # res = coupling_variables(x+k) - coupling_variables(x)
            res = new_couplings - current_couplings

            # compute_normed_residual
            self._compute_residual(
                current_couplings, new_couplings, current_iter, first=True, log_normed_residual=self.log_convergence
            )

            if self._stop_criterion_is_reached:
                break
            # compute newton step with res and gradients computed with x=
            # coupling_variables(n)
            newton_step_dict = self.assembly.compute_newton_step_pure(
                res,
                self.strong_couplings,
                self.over_relaxation_factor,
                self.linear_solver,
                matrix_type=self.matrix_type,
                **self.linear_solver_options,
            )

            # ynew = yk+1 + step
            # update current solution with Newton step
            # we update coupling_variables(n) with newton step
            for c_var, c_step in newton_step_dict.items():
                old_x_array[c_var] += c_step.real  # SoSTrades fix (.real)

            # convert old_x_array into SoSTrades types and store it into local_data for next execution
            self.local_data.update(convert_array_into_new_type(old_x_array, self._disciplines[0].reduced_dm))

            # store current_couplings for residual computation of next iteration
            current_couplings = np.hstack(list(old_x_array.values()))
            current_iter += 1

    def _current_strong_couplings(self, return_converted_dict=False):  # type: (...) -> ndarray
        """Return the current values of the strong coupling variables."""
        if len(self.strong_couplings) == 0:
            return np.array([])
        strong_couplings_array = {}
        # build a dictionary of strong_couplings values
        for input_key in self.strong_couplings:
            strong_couplings_array[input_key], new_dm = convert_new_type_into_array(
                input_key, self.local_data[input_key], self.disciplines[0].reduced_dm
            )
        # concatenate strong_couplings values
        concat_strong_couplings = np.hstack(list(strong_couplings_array.values()))

        if return_converted_dict:
            return concat_strong_couplings, strong_couplings_array
        return concat_strong_couplings
