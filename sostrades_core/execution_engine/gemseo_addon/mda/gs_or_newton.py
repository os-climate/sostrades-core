'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/04-2024/05/16 Copyright 2023 Capgemini

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
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from gemseo.core.execution_status import ExecutionStatus
from gemseo.mda.sequential_mda import MDAGSNewton, MDASequential

from sostrades_core.execution_engine.gemseo_addon.mda.gauss_seidel import (
    SoSMDAGaussSeidel,
)
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline

if TYPE_CHECKING:
    from gemseo.core.discipline.discipline import Discipline

LOGGER = logging.getLogger("gemseo.addons.mda.gs_or_newton")


class GSorNewtonMDA(MDASequential):
    """
    Perform some GaussSeidel iterations and then NewtonRaphson iterations.
    A chain of MDAs to build hybrids of MDA algorithms sequentially

    """

    def __init__(
        self,
        disciplines: Sequence[Discipline],
        name: str | None = None,
        grammar_type: str = ProxyDiscipline.SOS_GRAMMAR_TYPE,
        tolerance: float = 1e-6,
        max_mda_iter: int = 10,
        over_relaxation_factor: float = 0.99,
        linear_solver: str = "lgmres",
        tolerance_gs: float = 10.0,
        max_mda_iter_gs: int = 10,
        linear_solver_tolerance: float = 1e-12,
        scaling_method: MDASequential.ResidualScaling = MDASequential.ResidualScaling.N_COUPLING_VARIABLES,
        linear_solver_options: Mapping[str, Any] | None = None,
        warm_start: bool = False,
        use_lu_fact: bool = False,
        **newton_mda_options,
    ) -> None:
        """
        Constructor

        :param disciplines: the disciplines list
        :type disciplines: list(Discipline)
        :param name: name
        :type name: str
        :param grammar_type: the type of grammar to use for IO declaration
            either JSON_GRAMMAR_TYPE or SIMPLE_GRAMMAR_TYPE
        :type grammar_type: str
        :param tolerance: tolerance of the iterative direct coupling solver,
            norm of the current residuals divided by initial residuals norm
            shall be lower than the tolerance to stop iterating
        :type tolerance: float
        :param max_mda_iter: maximum number of iterations
        :type max_mda_iter: int
        :param relax_factor: relaxation factor
        :type relax_factor: float
        :param linear_solver: type of linear solver to be used to solve
            the Newton problem
        :type linear_solver: str
        :param max_mda_iter_gs: maximum number of iterations of the GaussSeidel
            solver
        :type max_mda_iter_gs: int
        :param warm_start: if True, the second iteration and ongoing
            start from the previous coupling solution
        :type warm_start: bool
        :param linear_solver_tolerance: Tolerance of the linear solver
            in the adjoint equation
        :type linear_solver_tolerance: float
        :param use_lu_fact: if True, when using adjoint/forward
            differenciation, store a LU factorization of the matrix
            to solve faster multiple RHS problem
        :type use_lu_fact: bool
        :param newton_mda_options: options passed to the MDANewtonRaphson
        :type newton_mda_options: dict
        """
        mda_gs = SoSMDAGaussSeidel(disciplines, max_mda_iter=max_mda_iter_gs, name=f'{name}_MDAGS',
                                   grammar_type=grammar_type,
                                   tolerance=tolerance)

        mda_newton = MDAGSNewton(
            disciplines,
            max_mda_iter=max_mda_iter,
            name=f'{name}_MDAGSNewton',
            linear_solver=linear_solver,
            linear_solver_options=linear_solver_options,
            use_lu_fact=use_lu_fact,
            tolerance=tolerance,
            over_relaxation_factor=over_relaxation_factor,
            **newton_mda_options,
        )
        # set the tolerance for the GS MDA
        mda_newton.mda_sequence[0].tolerance = tolerance_gs

        sequence = [mda_gs, mda_newton]
        super().__init__(
            disciplines,
            sequence,
            name=name,
            max_mda_iter=max_mda_iter,
            tolerance=tolerance,
            linear_solver_options=linear_solver_options,
            linear_solver_tolerance=linear_solver_tolerance,
            warm_start=warm_start,
        )

        # set the residual scaling method
        self.scaling = scaling_method

    def _run(self):
        """Runs the MDAs in a sequential way

        :returns: the local data
        """
        self._couplings_warm_start()
        # execute MDAs in sequence
        if self.reset_history_each_run:
            self.residual_history = []
        # initialize dm_values
        dm_values = {}
        try:
            mda_i = self.mda_sequence[1]
            mda_i.execution_status.value = ExecutionStatus.Status.PENDING
            # TODO: [discuss limitations] mechanism not possible in EEV4 --> remove
            # dm_values = deepcopy(self.disciplines[0].dm.get_data_dict_values())

            self.io.data = mda_i.execute(self.io.data)
        except:
            LOGGER.warning('The MDAGSNewton has not converged try with MDAGaussSeidel')
            mda_i = self.mda_sequence[0]
            mda_i.execution_status.value = ExecutionStatus.Status.PENDING

            # TODO: [discuss limitations] mechanism not possible in EEV4 --> remove
            # dm = self.disciplines[0].ee.dm
            # # set values directrly in dm to avoid reconfigure of disciplines
            # dm.set_values_from_dict(dm_values)
            # self.disciplines[0].ee.load_study_from_input_dict(dm_values)
            self.io.data = mda_i.execute(self.io.data)

        self.residual_history += mda_i.residual_history
