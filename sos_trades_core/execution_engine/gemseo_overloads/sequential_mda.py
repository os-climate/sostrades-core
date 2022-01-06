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
# pylint: skip-file
from gemseo.mda.sequential_mda import MDASequential, GSNewtonMDA
from gemseo.core.discipline import MDODiscipline
from sos_trades_core.execution_engine.gemseo_addon.mda.gauss_seidel import SoSMDAGaussSeidel  # pylint: disable-msg=E0611,E0001
from sos_trades_core.execution_engine.gemseo_overloads.newton import MDANewtonRaphson  # pylint: disable-msg=E0611


def __init__(
    cls,
    disciplines,  # type: Sequence[MDODiscipline]
    name=None,  # type: Optional[str]
    grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,  # type: str
    tolerance=1e-6,  # type: float
    max_mda_iter=10,  # type: int
    relax_factor=0.99,  # type: float
    linear_solver="DEFAULT",  # type: str
    max_mda_iter_gs=3,  # type: int
    linear_solver_tolerance=1e-12,  # type: float
    warm_start=False,  # type: bool
    use_lu_fact=False,  # type: bool
    coupling_structure=None,  # type: Optional[MDOCouplingStructure]
    linear_solver_options=None,  # type: Mapping[str,Any]
    log_convergence=False,  # type: bool
    **newton_mda_options  # type: float
):
    """
    Args:
        relax_factor: The relaxation factor.
        linear_solver: The type of linear solver
            to be used to solve the Newton problem.
        max_mda_iter_gs: The maximum number of iterations
            of the Gauss-Seidel solver.
        log_convergence: Whether to log the MDA convergence,
            expressed in terms of normed residuals.
        **newton_mda_options: The options passed to :class:`.MDANewtonRaphson`.
    """
    mda_gs = SoSMDAGaussSeidel(
        disciplines,
        max_mda_iter=max_mda_iter_gs,
        name=None,
        grammar_type=grammar_type,
        log_convergence=log_convergence,
    )
    mda_gs.tolerance = tolerance
    mda_newton = MDANewtonRaphson(
        disciplines,
        max_mda_iter,
        relax_factor,
        tolerance=tolerance,
        name=None,
        grammar_type=grammar_type,  # SoSTrades fix
        linear_solver=linear_solver,
        use_lu_fact=use_lu_fact,
        coupling_structure=coupling_structure,
        log_convergence=log_convergence,
        linear_solver_options=linear_solver_options,
        linear_solver_tolerance=linear_solver_tolerance,
        ** newton_mda_options
    )
    sequence = [mda_gs, mda_newton]
    super(GSNewtonMDA, cls).__init__(
        disciplines,
        sequence,
        name=name,
        grammar_type=grammar_type,
        max_mda_iter=max_mda_iter,
        tolerance=tolerance,
        linear_solver_tolerance=linear_solver_tolerance,
        warm_start=warm_start,
        linear_solver=linear_solver,
        linear_solver_options=linear_solver_options,
        coupling_structure=coupling_structure,
    )


# Set functions to the MDA Class
setattr(GSNewtonMDA, "__init__", __init__)
