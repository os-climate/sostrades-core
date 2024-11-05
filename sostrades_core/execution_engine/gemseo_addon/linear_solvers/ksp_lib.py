'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/09-2024/06/24 Copyright 2023 Capgemini

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
from typing import TYPE_CHECKING, Any, ClassVar

import petsc4py
from gemseo.algos.linear_solvers.base_linear_solver_library import (
    BaseLinearSolverLibrary,
    LinearSolverDescription,
)
from gemseo_petsc.linear_solvers.petsc_ksp import (
    _convert_ndarray_to_mat_or_vec,
)
from gemseo_petsc.linear_solvers.settings.petsc_ksp_settings import BaseSoSPetscKSPSettings
from numpy import isnan, ndarray

# Must be done before from petsc4py import PETSc, this loads the options from command args in the options database.
petsc4py.init([])
from petsc4py import PETSc  # noqa: E402

if TYPE_CHECKING:
    from gemseo.algos.linear_solvers.linear_problem import LinearProblem

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
KSP_CONVERGED_REASON = {
    1: 'KSP_CONVERGED_RTOL_NORMAL',
    9: 'KSP_CONVERGED_ATOL_NORMAL',
    2: 'KSP_CONVERGED_RTOL',
    3: 'KSP_CONVERGED_ATOL',
    4: 'KSP_CONVERGED_ITS',
    5: 'KSP_CONVERGED_CG_NEG_CURVE',
    6: 'KSP_CONVERGED_CG_CONSTRAINED',
    7: 'KSP_CONVERGED_STEP_LENGTH',
    8: 'KSP_CONVERGED_HAPPY_BREAKDOWN',
    -2: 'KSP_DIVERGED_NULL',
    -3: 'KSP_DIVERGED_ITS',
    -4: 'KSP_DIVERGED_DTOL',
    -5: 'KSP_DIVERGED_BREAKDOWN',
    -6: 'KSP_DIVERGED_BREAKDOWN_BICG',
    -7: 'KSP_DIVERGED_NONSYMMETRIC',
    -8: 'KSP_DIVERGED_INDEFINITE_PC',
    -9: 'KSP_DIVERGED_NANORINF',
    -10: 'KSP_DIVERGED_INDEFINITE_MAT',
    -11: 'KSP_DIVERGED_PC_FAILED',
    0: 'KSP_CONVERGED_ITERATING',
}


class SoSPetscKSPAlgos(BaseLinearSolverLibrary):
    """Interface to PETSC KSP.

    For further information, please read
    https://petsc4py.readthedocs.io/en/stable/manual/ksp/
    https://petsc.org/release/docs/manualpages/KSP/KSP.html#KSP
    https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.KSP.Type-class.html

    KSP example here:
    https://fossies.org/linux/petsc/src/binding/petsc4py/demo/petsc-examples/ksp/ex2.py
    """

    AVAILABLE_LINEAR_SOLVERS: tuple[str] = ('GMRES_PETSC', 'LGMRES_PETSC', 'BICG_PETSC', 'BCGS_PETSC')
    """The available linear solvers."""

    AVAILABLE_PRECONDITIONERS: tuple[str] = ('jacobi', 'ilu', 'gasm')
    """The available preconditioners."""

    ALGORITHM_INFOS: ClassVar[dict[str, LinearSolverDescription]] = {
        solver_name: LinearSolverDescription(
            algorithm_name=solver_name,
            description="Linear solver " + solver_name,
            internal_algorithm_name=solver_name,
            lhs_must_be_linear_operator=True,
            library_name="PETSC_KSP",
            website="https://petsc.org/release/docs/manualpages/KSP/KSP.html#KSP",
            Settings=type(
                f"Petsc{solver_name}Settings",
                (BaseSoSPetscKSPSettings,),
                {"_TARGET_CLASS_NAME": f"{solver_name}_PETSC"},
            ),
        )
        for solver_name in AVAILABLE_LINEAR_SOLVERS
    }

    LOGGER.debug('algos infos: %s', ALGORITHM_INFOS)

    def _run(self, problem: LinearProblem, **settings: Any) -> ndarray:
        """Run the algorithm.

        Args:
            problem: The linear problem to solve.
            **settings: The algorithm settings.

        Returns:
            The solution of the problem.
        """
        # set default settings
        # TODO: move to main linear solver classes so that all solvers benefit
        # from these default inputs definition
        b = problem.rhs
        a = problem.lhs
        if 'maxiter' not in settings:
            settings['maxiter'] = 50 * b.shape[0]
        else:
            settings['maxiter'] = min(settings['maxiter'], 50 * a.shape[0])

        # first run
        settings["old_sol"] = None

        sol, info, ksp = self._run_petsc_strategy(problem, **settings)

        if info < 0:
            settings['solver_type'] = 'bcgs'
            settings['preconditioner_type'] = 'gasm'
            settings['old_sol'] = sol

            # second run
            sol, info, ksp = self._run_petsc_strategy(problem, **settings)
            if info >= 0:
                LOGGER.warning(
                    "The second try with GASM preconditioner and bi CG stabilized linear solver has converged at %s",
                    ksp.getResidualNorm(),
                )
            elif info == -3:
                # DIVERGED_ITS
                LOGGER.warning(
                    "DIVERGED_ITS error : the number of iterations of the solver is %s with a max iter of %s, running again with 10*max_iter",
                    len(ksp.getConvergenceHistory()),
                    settings['maxiter'],
                )
                settings['maxiter'] = 10 * settings['maxiter']
                settings['solver_type'] = 'bcgs'
                settings['preconditioner_type'] = 'gasm'
                settings['old_sol'] = sol

                # third run
                sol, info, ksp = self._run_petsc_strategy(problem, **settings)

        return problem.solution

    def _run_petsc_strategy(self, problem, **settings):
        # Initialize the KSP solver.
        options_cmd = settings.get("options_cmd")
        if options_cmd is not None:
            petsc4py.init(options_cmd)
        else:
            petsc4py.init()

        # Create the solver
        ksp = PETSc.KSP().create()
        # Set all solver settings
        ksp.setType(settings["solver_type"])
        ksp.setTolerances(settings["rtol"], settings["atol"], settings["dtol"], settings["maxiter"])
        ksp.setConvergenceHistory()

        b = problem.rhs
        a = problem.lhs

        # CHeck Nan in matrix
        if isnan(a.min()):
            LOGGER.error('The matrix A in the linear solver contains a NaN')
        # Transform sparse matrix into petsc matrix
        a_mat = _convert_ndarray_to_mat_or_vec(a)
        # Set the matrix in the solver
        ksp.setOperators(a_mat)
        # the chosen preconditioner
        prec_type = settings.get("preconditioner_type")
        if prec_type is not None:
            pc = ksp.getPC()
            pc.setType(prec_type)
            pc.setUp()

        # transform the b array in petsc vector
        b_mat = _convert_ndarray_to_mat_or_vec(b)
        # Use b as first solution (same size)
        old_sol = settings.get("old_sol")
        if old_sol is not None:
            solution = _convert_ndarray_to_mat_or_vec(old_sol)
        else:
            solution = b_mat.duplicate()
            solution.set(0)
        if settings["view_config"]:
            ksp.view()
        # Solve the ksp petsc solver
        ksp.solve(b_mat, solution)
        problem.solution = solution.getArray().copy()  # added a copy() like in GEMSEO
        problem.convergence_info = ksp.reason

        # Reason to positive for convergence, 0 for no convergence, and negative for failure to converge
        if problem.convergence_info > 0:
            info = 0
            if problem.convergence_info in (1, 2):
                LOGGER.warning(
                    'The PETSc linear solver has converged with relative tolerance %s, the final residual norm is %s ; check your linear problem',
                    settings["tol"],
                    ksp.getResidualNorm(),
                )
            elif problem.convergence_info == 4:
                LOGGER.warning(
                    'The PETSc linear solver has converged after max iterations %s, the final residual norm is %s ; check your linear problem',
                    settings["maxiter"],
                    ksp.getResidualNorm(),
                )
            elif problem.convergence_info in (3, 9):
                pass
            else:
                LOGGER.warning(
                    'The PETSc linear solver has converged with %s, the tolerance is %s, the final residual norm is %s ; check your linear problem',
                    KSP_CONVERGED_REASON[problem.convergence_info],
                    settings["atol"],
                    ksp.getResidualNorm(),
                )
        elif problem.convergence_info == 0:
            info = 1
        else:
            info = problem.convergence_info
            LOGGER.warning(
                'The PETSc linear solver has not converged with error %s, the final residual norm is %s ; check your linear problem',
                KSP_CONVERGED_REASON[problem.convergence_info],
                ksp.getResidualNorm(),
            )
            LOGGER.warning(
                'The convergence_history of length %s is %s',
                len(ksp.getConvergenceHistory()),
                ksp.getConvergenceHistory(),
            )
        petsc4py.PETSc.garbage_cleanup()
        return problem.solution, info, ksp
