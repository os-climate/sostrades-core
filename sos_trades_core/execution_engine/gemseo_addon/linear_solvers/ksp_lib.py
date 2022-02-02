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
"""A PETSC KSP linear solvers library wrapper."""
import logging
import sys
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import petsc4py  # pylint: disable-msg=E0401
from gemseo.algos.linear_solvers.linear_solver_lib import LinearSolverLib
from gemseo_petsc.linear_solvers.ksp_lib import _convert_ndarray_to_mat_or_vec  # pylint: disable-msg=E0401
from numpy import arange
from numpy import array
from numpy import ndarray
from numpy import isnan
from scipy.sparse import csr_matrix
from scipy.sparse import find
from scipy.sparse.base import issparse

# Must be done before from petsc4py import PETSc, this loads the options from
# command args in the options database.
petsc4py.init(sys.argv)
from petsc4py import PETSc  # pylint: disable-msg=E0401

LOGGER = logging.getLogger(__name__)

KSP_CONVERGED_REASON = {1: 'KSP_CONVERGED_RTOL_NORMAL',
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
                        0: 'KSP_CONVERGED_ITERATING'}

# TODO: inherit from PetscKSPAlgo of GEMSEO


class PetscKSPAlgos(LinearSolverLib):
    """Interface to PETSC KSP.

    For further information, please read
    https://petsc4py.readthedocs.io/en/stable/manual/ksp/

    https://petsc.org/release/docs/manualpages/KSP/KSP.html#KSP
    """
    # https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.KSP.Type-class.html
    AVAILABLE_LINEAR_SOLVER = ['GMRES_PETSC',
                               'LGMRES_PETSC', 'BICG_PETSC', 'BCGS_PETSC']
    AVAILABLE_PRECONDITIONER = ['jacobi', 'ilu', 'gasm']

    def __init__(self):
        super().__init__()

        self.lib_dict = {name: self.get_default_properties(
            name) for name in self.AVAILABLE_LINEAR_SOLVER}

        self.default_tol = 1e-200

    @classmethod
    def get_default_properties(cls, algo_name):
        """Return the properties of the algorithm.
        It states if it requires symmetric,
        or positive definite matrices for instance.
        Args:
            algo_name: The algorithm name.
        Returns:
            The properties of the solver.
        """
        return {cls.LHS_MUST_BE_POSITIVE_DEFINITE: False,
                cls.LHS_MUST_BE_SYMMETRIC: False,
                cls.LHS_CAN_BE_LINEAR_OPERATOR: True,
                cls.INTERNAL_NAME: algo_name}

    def _get_options(
        self,
        solver_type="gmres",  # type: str
        max_iter=100000,  # type: int
        tol=1.0e-200,  # type: float
        atol=1e-8,  # type: float
        dtol=1.0e50,  # type: float
        preconditioner_type="ilu",  # type: str
        view_config=False,  # type: bool
        ksp_pre_processor=None,  # type: Optional[bool]
        options_cmd=None,  # type: Optional[Dict[str, Any]]
        set_from_options=False,  # type: bool
        monitor_residuals=False,  # type: bool
    ):  # type: (...) -> Dict[str, Any]
        """Return the algorithm options.

        This method returns the algoritms options after having done some checks,
        and if necessary,
        set the default values.

        Args:
            solver_type: The KSP solver type.
                See `https://petsc.org/release/docs/manualpages/KSP/KSPType.html#KSPType`_
            max_iter: The maximum number of iterations.
            tol: The relative convergence tolerance,
                relative decrease in the (possibly preconditioned) residual norm.
            atol: The absolute convergence tolerance of the
                (possibly preconditioned) residual norm.
            dtol: The divergence tolerance,
                e.g. the amount the (possibly preconditioned) residual norm can increase.
            preconditioner_type: The type of the precondtioner,
                see `https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.PC.Type-class.html`_ # noqa: B950
            view_config: Whether to call ksp.view() to view the configuration
                of the solver before run.
            ksp_pre_processor: A callback function that is called with (KSP problem,
                options dict) as arguments before calling ksp.solve().
                It allows the user to obtain an advanced configuration that is not
                supported by the current wrapper.
                If None, do not perform any call.
            options_cmd: The options to pass to the PETSc KSP solver.
                If None, use the default options.
            set_from_options: Whether the options are set from sys.argv,
                a classical Petsc configuration mode.
            monitor_residuals: Whether to store the residuals during convergence.
                WARNING: as said in Petsc documentation,
                 "the routine is slow and should be used only for
                 testing or convergence studies, not for timing."

        Returns:
            The algorithm options.
        """
        return self._process_options(
            max_iter=max_iter,
            solver_type=solver_type,
            monitor_residuals=monitor_residuals,
            tol=tol,
            atol=atol,
            dtol=dtol,
            preconditioner_type=preconditioner_type,
            view_config=view_config,
            options_cmd=options_cmd,
            set_from_options=set_from_options,
            ksp_pre_processor=ksp_pre_processor,
        )

    def __monitor(
        self,
        ksp,  # type: PETSc.KSP
        its,  # type: int
        rnorm,  # type: float
    ):  # type: (...) -> None
        """Add the normed residual value to the problem residual history.

        This method is aimed to be passed to petsc4py as a reference.
        This is the reason why some of its arguments are not used.

        Args:
             ksp: The KSP PETSc solver.
             its: The current iteration.
             rnorm: The normed residual.
        """
        self.problem.residuals_history.append(rnorm)

    def _run(
        self, **options  # type: Any
    ):  # type: (...) -> ndarray
        """Run the algorithm.

        Args:
            **options: The algorithm options.

        Returns:
            The solution of the problem.
        """
        # set default options
        # TODO: move to main linear solver classes so that all solvers benefit
        # from these default inputs definition
        options['max_iter'] = int(options['max_iter'])
#         if 'tol' not in options:
#             options['tol'] = 1e-8
        options['atol'] = options['tol']
        options['tol'] = self.default_tol
        b = self.problem.rhs
        A = self.problem.lhs
        if 'maxiter' not in options:
            options['maxiter'] = 50 * b.shape[0]
        else:
            options['maxiter'] = min(
                options['maxiter'], 50 * A.shape[0])

        # first run
        options["old_sol"] = None
        LOGGER.info(f'PETSC options : {options}')
        sol, info, ksp = self._run_petsc_strategy(**options)

        if info < 0:

            options['solver_type'] = 'bcgs'
            options['preconditioner_type'] = 'gasm'
            options['old_sol'] = sol

            # second run
            sol, info, ksp = self._run_petsc_strategy(**options)

            if info >= 0:

                LOGGER.warning(
                    f'The second try with GASM preconditioner and bi CG stabilized linear solver has converged at {ksp.getResidualNorm()}')

            else:
                # compare the two tries and take the best one
                #                     if first_residual_error < ksp.getResidualNorm():
                #                         sol = firstsol
                #                         LOGGER.warning(
                #                             f'The first solve with {linear_solver_list[-2]} converges better than bi CG stabilized linear solver we take the first solution')
                #
                if info == -3:
                    # DIVERGED_ITS
                    LOGGER.warning(
                        f"DIVERGED_ITS error : the number of iterations of the solver is {len(ksp.getConvergenceHistory())} with a max iter of {options['maxiter']}, try to launch again with 10*max_iter")
                    options['maxiter'] = 10 * options['maxiter']
                    options['solver_type'] = 'bcgs'
                    options['preconditioner_type'] = 'gasm'
                    options['old_sol'] = sol

                    # third run
                    sol, info, ksp = self._run_petsc_strategy(**options)

        return self.problem.solution

    def _run_petsc_strategy(self, **options):
        # Initialize the KSP solver.
        # Create the options database
        options_cmd = options.get("options_cmd")
        if options_cmd is not None:
            petsc4py.init(options_cmd)
        else:
            petsc4py.init()

        # Create the solver
        ksp = PETSc.KSP().create()
        # Set all solver options
        ksp.setType(options["solver_type"])
        ksp.setTolerances(
            options["tol"], options["atol"], options["dtol"], options["max_iter"]
        )
        ksp.setConvergenceHistory()

        b = self.problem.rhs
        A = self.problem.lhs

        # CHeck Nan in matrix
    #     LOGGER.info('check if A contains a NaN')
        if isnan(A.min()):
            LOGGER.error('The matrix A in the linear solver contains a NaN')
        # Transform sparse matrix into petsc matrix
        a_mat = _convert_ndarray_to_mat_or_vec(A)
        # Set the matrix in the solver
        ksp.setOperators(a_mat)
        # the chosen preconditioner
        prec_type = options.get("preconditioner_type")
        if prec_type is not None:
            pc = ksp.getPC()
            pc.setType(prec_type)
            pc.setUp()

#         # Allow for solver choice to be set from command line with -ksp_type <solver>.
#         # Recommended option: -ksp_type preonly -pc_type lu
#         if options["set_from_options"]:
#             ksp.setFromOptions()
#
#         ksp_pre_processor = options.get("ksp_pre_processor")
#         if ksp_pre_processor is not None:
#             ksp_pre_processor(ksp, options)
#
#         self.problem.residuals_history = []
#         if options["monitor_residuals"]:
#             LOGGER.warning(
#                 "Petsc option monitor_residuals slows the process and"
#                 " should be used only for testing or convergence studies."
#             )
#             ksp.setMonitor(self.__monitor)

        # transform the b array in petsc vector
        b_mat = _convert_ndarray_to_mat_or_vec(b)
        # Use b as first solution (same size)
        old_sol = options.get("old_sol")
        if old_sol is not None:
            solution = _convert_ndarray_to_mat_or_vec(old_sol)
        else:
            solution = b_mat.duplicate()
            solution.set(0)
        if options["view_config"]:
            ksp.view()
        # Solve the ksp petsc solver
        ksp.solve(b_mat, solution)
        sol = solution.getArray().copy()  # added a copy() like in GEMSEO
        convergence_info = ksp.reason

        # update of problem attributes
        self.problem.solution = sol
        self.problem.convergence_info = ksp.reason

    #         method_list = [func for func in dir(ksp) if callable(getattr(ksp, func))]
    #         print(method_list)

    #     print('it_number', ksp.getIterationNumber())
    #     print('residual_norm', ksp.getResidualNorm())
    #     print('convergence_info', convergence_info)

        # reason to positive for convergence, 0 for no convergence, and negative
        # for failure to converge
        if convergence_info > 0:
            info = 0

            if convergence_info == 2 or convergence_info == 1:
                LOGGER.warning(
                    f'The PETSc linear solver has converged with relative tolerance {options["tol"]}, the final residual norm is {ksp.getResidualNorm()} check your linear problem')
            elif convergence_info == 4:
                LOGGER.warning(
                    f'The PETSc linear solver has converged after max iterations {options["max_iter"]}, the final residual norm is {ksp.getResidualNorm()} check your linear problem')
            elif convergence_info == 3 or convergence_info == 9:
                pass
            else:
                LOGGER.warning(
                    f'The PETSc linear solver has converged with {KSP_CONVERGED_REASON[convergence_info]}, the final residual norm is {ksp.getResidualNorm()} check your linear problem')
        elif convergence_info == 0:
            info = 1
        else:
            info = convergence_info
            LOGGER.warning(
                f'The PETSc linear solver has not converged with error {KSP_CONVERGED_REASON[convergence_info]}, the final residual norm is {ksp.getResidualNorm()} check your linear problem')
            LOGGER.warning(
                f' The convergence_history of length {len(ksp.getConvergenceHistory())} is {ksp.getConvergenceHistory()}')

        return sol, info, ksp


# KSP example here
# https://fossies.org/linux/petsc/src/binding/petsc4py/demo/petsc-examples/ksp/ex2.py
