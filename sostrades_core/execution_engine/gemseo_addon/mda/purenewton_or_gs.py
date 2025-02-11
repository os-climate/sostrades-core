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

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from gemseo import create_mda
from gemseo.mda.sequential_mda import MDASequential

from sostrades_core.execution_engine.gemseo_addon.mda.gauss_seidel import SOS_GRAMMAR_TYPE, SoSMDAGaussSeidel

if TYPE_CHECKING:
    from gemseo.core.discipline.discipline import Discipline

LOGGER = logging.getLogger("gemseo.addons.mda.purenewton_or_gs")


class GSPureNewtonorGSMDA(MDASequential):
    """Perform some GaussSeidel iterations and then NewtonRaphson iterations."""

    def __init__(
        self,
        disciplines: Sequence[Discipline],
        name: str | None = None,
        grammar_type: str = SOS_GRAMMAR_TYPE,
        tolerance: float = 1e-6,
        max_mda_iter: int = 10,
        relax_factor: float = 0.99,
        linear_solver: str = "DEFAULT",
        tolerance_gs: float = 10.0,
        max_mda_iter_gs: int = 10,
        linear_solver_tolerance: float = 1e-12,
        scaling_method: MDASequential.ResidualScaling = MDASequential.ResidualScaling.N_COUPLING_VARIABLES,
        warm_start: bool = False,
        use_lu_fact: bool = False,
        linear_solver_settings: Mapping[str, Any] | None = None,
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
        mda_gs = SoSMDAGaussSeidel(
            disciplines, max_mda_iter=max_mda_iter_gs, tolerance=tolerance_gs, name=None, grammar_type=grammar_type
        )
        mda_gs.tolerance = tolerance

        mda_newton = create_mda(
            'GSPureNewtonMDA',
            disciplines,
            max_mda_iter=max_mda_iter,
            name=None,
            linear_solver=linear_solver,
            linear_solver_settings=linear_solver_settings,
            tolerance_gs=tolerance_gs,
            max_mda_iter_gs=max_mda_iter_gs,
            use_lu_fact=use_lu_fact,
            tolerance=tolerance,
            relax_factor=relax_factor,
            **newton_mda_options,
        )

        sequence = [mda_gs, mda_newton]
        super().__init__(
            disciplines,
            sequence,
            name=name,
            max_mda_iter=max_mda_iter,
            tolerance=tolerance,
            linear_solver_settings=linear_solver_settings,
            linear_solver_tolerance=linear_solver_tolerance,
            warm_start=warm_start,
        )
        self.scaling = scaling_method

    def _execute(self):
        """
        Runs the MDAs in a sequential way

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
            mda_i.reset_statuses_for_run()
            dm_values = deepcopy(self._disciplines[0].dm.get_data_dict_values())
            self.io.data = mda_i.execute(self.io.data)
        except:
            LOGGER.warning('The GSPureNewtonMDA has not converged try with MDAGaussSeidel')
            mda_i = self.mda_sequence[0]
            mda_i.reset_statuses_for_run()
            dm = self._disciplines[0].ee.dm
            # set values directrly in dm to avoid reconfigure of disciplines
            dm.set_values_from_dict(dm_values)
            # self.disciplines[0].ee.load_study_from_input_dict(dm_values)
            self.io.data = mda_i.execute(self.io.data)

        self.residual_history += mda_i.residual_history
