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
from gemseo.mda.newton import MDANewtonRaphson, MDARoot
from copy import deepcopy
from sos_trades_core.execution_engine.parallel_execution.sos_parallel_execution import SoSDiscParallelExecution
from gemseo.core.discipline import MDODiscipline


def __init__(
    cls,
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
    cls.n_processes = n_processes

    super(MDANewtonRaphson, cls).__init__(
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
    #cls.relax_factor = cls.__check_relax_factor(relax_factor)
    cls.relax_factor = relax_factor
    cls.linear_solver = linear_solver

    # Add parallel execution for NewtonRaphson

    cls.parallel_execution = SoSDiscParallelExecution(
        disciplines, n_processes=cls.n_processes, use_threading=True
    )
    # end of SoSTrades modification


def execute_all_disciplines(
    cls,
    input_local_data,  # type: Mapping[str,ndarray]
):  # type: (...) -> None
    """Execute all the disciplines.
        Come from MDAJacobi
    Args:
        input_local_data: The input data of the disciplines.
    """
    cls.reset_disciplines_statuses()

    if hasattr(cls, 'n_processes') and cls.n_processes > 1:
        n_disc = len(cls.disciplines)
        inputs_copy_list = [deepcopy(input_local_data) for _ in range(n_disc)]
        cls.parallel_execution.execute(inputs_copy_list)
    else:
        for disc in cls.disciplines:
            disc.reset_statuses_for_run()
            disc.execute(deepcopy(input_local_data))

    outputs = [discipline.get_output_data() for discipline in cls.disciplines]
    for data in outputs:
        cls.local_data.update(data)


def _newton_step(cls):  # type: (...) -> None
    """Execute the full Newton step.

    Compute the increment :math:`-[dR/dW]^{-1}.R` and run the disciplines.
    """
    # SoSTrades fix: pass linear solver tolerance to linear_solver_options...
    tol_dict = {'tol': cls.linear_solver_tolerance}
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
        exec_data[c_var] += c_step.real  # SoSTrades fix (.real)
    cls.reset_disciplines_statuses()
    cls.execute_all_disciplines(exec_data)


# Set functions to the MDA Class
setattr(MDANewtonRaphson, "__init__", __init__)
setattr(MDANewtonRaphson, "_newton_step", _newton_step)
setattr(MDARoot, "execute_all_disciplines", execute_all_disciplines)
