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
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from gemseo.mda.sequential_mda import MDASequential

from sostrades_core.execution_engine.gemseo_addon.mda.gauss_seidel import SOS_GRAMMAR_TYPE, SoSMDAGaussSeidel
from sostrades_core.execution_engine.gemseo_addon.mda.pure_newton_raphson import (
    PureNewtonRaphson,
)

if TYPE_CHECKING:
    from gemseo.core.coupling_structure import CouplingStructure
    from gemseo.core.discipline.discipline import Discipline

LOGGER = logging.getLogger("gemseo.addons.mda.gs_purenewton_mda")


class GSPureNewtonMDA(MDASequential):
    """Perform some GaussSeidel iterations and then PureNewtonRaphson iterations."""

    def __init__(
        self,
        disciplines: Sequence[Discipline],
        name: str | None = None,
        grammar_type: str = SOS_GRAMMAR_TYPE,
        tolerance: float = 1e-6,
        max_mda_iter: int = 10,
        over_relaxation_factor: float = 0.99,
        linear_solver: str = "DEFAULT",
        tolerance_gs: float = 10.0,
        max_mda_iter_gs: int = 10,
        linear_solver_tolerance: float = 1e-12,
        scaling_method: MDASequential.ResidualScaling = MDASequential.ResidualScaling.N_COUPLING_VARIABLES,
        warm_start: bool = False,
        use_lu_fact: bool = False,
        coupling_structure: CouplingStructure = None,
        linear_solver_settings: Mapping[str, Any] | None = None,
        log_convergence: bool = False,
        **newton_mda_options,
    ):
        """
        Args:
            disciplines: set of disciplines in the MDA
            name: name of the MDA
            grammar_type: type of grammar used
            tolerance: target maximum residual at convergence
            max_mda_iter: maximum of iterations of the MDA that can be conducted
            over_relaxation_factor: the over relaxation factor in the Newton step.
            linear_solver: The type of linear solver to be used to solve the Newton problem.
            tolerance_gs: target maximum residual of the Gauss-Seidel strategy
            max_mda_iter_gs: maximum number of iterations of the Gauss-Seidel solver
            linear_solver_tolerance: tolerance for linear solver
            scaling_method: scaling method applied for computation of residual
            warm_start: wether to use warm start
            use_lu_fact: whether to use LU factorization
            coupling_structure: coupling structure
            linear_solver_settings: settings for linear solver
            log_convergence: Whether to log the MDA convergence, expressed in terms of normed residuals.
            **newton_mda_options: The options passed to :class:`.MDANewtonRaphson`.

        """
        mda_gs = SoSMDAGaussSeidel(
            disciplines, max_mda_iter=max_mda_iter_gs, name=None, grammar_type=grammar_type, tolerance=tolerance_gs
        )

        mda_newton = PureNewtonRaphson(
            disciplines,
            max_mda_iter,
            over_relaxation_factor,
            tolerance=tolerance,
            name=None,
            grammar_type=grammar_type,  # SoSTrades fix
            linear_solver=linear_solver,
            use_lu_fact=use_lu_fact,
            coupling_structure=coupling_structure,
            log_convergence=log_convergence,
            linear_solver_settings=linear_solver_settings,
            linear_solver_tolerance=linear_solver_tolerance,
            **newton_mda_options,
        )

        sequence = [mda_gs, mda_newton]
        super().__init__(
            disciplines,
            sequence,
            name=name,
            max_mda_iter=max_mda_iter,
            tolerance=tolerance,
            linear_solver_tolerance=linear_solver_tolerance,
            warm_start=warm_start,
            linear_solver=linear_solver,
            linear_solver_settings=linear_solver_settings,
            coupling_structure=coupling_structure,
        )
        self.scaling = scaling_method

    def _execute(self):
        """
        Override _run of sequential MDA to update PureNR MDA local data and normed residual
        with the values from GS MDA, to avoid an early termination flag before residual
        recalculation
        """
        self._couplings_warm_start()
        # execute MDAs in sequence
        if self.reset_history_each_run:
            self.residual_history = []
        for mda_i in self.mda_sequence:
            mda_i.reset_statuses_for_run()
            if mda_i.name == 'PureNewtonRaphson':
                mda_i.io.data = self.mda_sequence[0].io.data
                mda_i.normed_residual = self.mda_sequence[0].normed_residual
            self.io.data = mda_i.execute(self.io.data)
            self.residual_history += mda_i.residual_history
            if mda_i.normed_residual < self.tolerance:
                break
