'''
Copyright 2022 Airbus SAS
Modifications on 2023/03/27-2024/02/20 Copyright 2023 Capgemini

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
"""
Coupled derivatives calculations
********************************
"""

from collections import defaultdict
from numpy import empty, ones, zeros
from scipy.sparse import dia_matrix
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix
from os import getenv
from copy import deepcopy
from multiprocessing import Pool
import platform

from gemseo.core.jacobian_assembly import JacobianAssembly
from gemseo.algos.linear_solvers.linear_solvers_factory import LinearSolversFactory
from gemseo.algos.linear_solvers.linear_problem import LinearProblem

from sostrades_core.execution_engine.parallel_execution.sos_parallel_execution import SoSDiscParallelLinearization
from sostrades_core.tools.conversion.conversion_sostrades_sosgemseo import convert_new_type_into_array


def none_factory():
    """Returns None...

    To be used for defaultdict
    """


def default_dict_factory():
    """Instantiates a defaultdict(None) object."""
    return defaultdict(none_factory)


class SoSJacobianAssembly(JacobianAssembly):
    """Assembly of Jacobians Typically, assemble disciplines's Jacobians into a system
    Jacobian."""

    def __init__(self, coupling_structure, n_processes=1):
        self.n_processes = n_processes
        JacobianAssembly.__init__(self, coupling_structure)
        # Add parallel execution for NewtonRaphson

        self.parallel_linearize = SoSDiscParallelLinearization(
            self.coupling_structure.disciplines, n_processes=self.n_processes, use_threading=True)

    def _dres_dvar_sparse(self, residuals, variables, n_residuals, n_variables):
        """Forms the matrix of partial derivatives of residuals
        Given disciplinary Jacobians dYi(Y0...Yn)/dvj,
        fill the sparse Jacobian:
        |           |
        |  dRi/dvj  |
        |           |

        :param residuals: the residuals (R)
        :param variables: the differentiation variables
        :param n_residuals: number of residuals
        :param n_variables: number of variables
        """
        # SoSTrades modif
        dres_dvar = dok_matrix((n_residuals, n_variables))
        # end of SoSTrades modif

        out_i = 0
        # Row blocks
        for residual in residuals:
            residual_size = self.sizes[residual]
            # Find the associated discipline
            discipline = self._disciplines[residual]
            residual_jac = discipline.jac[residual]
            # Column blocks
            out_j = 0
            for variable in variables:
                variable_size = self.sizes[variable]
                if residual == variable:
                    # residual Yi-Yi: put -I in the Jacobian
                    ones_mat = (ones(variable_size), 0)
                    shape = (variable_size, variable_size)
                    diag_mat = -dia_matrix(ones_mat, shape=shape).tocoo()

                    if self.coupling_structure.is_self_coupled(discipline):
                        jac = residual_jac.get(variable, None)
                        if jac is not None:
                            diag_mat += jac
                    #                     dres_dvar[
                    #                         out_i: out_i + variable_size, out_j: out_j + variable_size
                    #                     ] = diag_mat
                    dict.update(dres_dvar,
                                {(out_i + jac_i, out_j + jac_j): jac_value for jac_i, jac_j, jac_value in
                                 zip(diag_mat.row.astype(float), diag_mat.col.astype(float), diag_mat.data)})
                else:
                    # block Jacobian
                    jac = residual_jac.get(variable, None)
                    if jac is not None:
                        n_i, n_j = jac.shape
                        assert n_i == residual_size
                        assert n_j == variable_size
                        jac = jac.tocoo()
                        # Fill the sparse Jacobian block
                        # dres_dvar[out_i: out_i + n_i, out_j: out_j + n_j] = jac
                        dict.update(dres_dvar,
                                    {(out_i + jac_i, out_j + jac_j): jac_value for jac_i, jac_j, jac_value in
                                     zip(jac.row.astype(float), jac.col.astype(float), jac.data)})

                # Shift the column by block width
                out_j += variable_size
            # Shift the row by block height
            out_i += residual_size
        return dres_dvar.tocsr()

    # SoSTrades modif
    def _dres_dvar_sparse_lil(self, residuals, variables, n_residuals, n_variables):
        """Forms the matrix of partial derivatives of residuals
        Given disciplinary Jacobians dYi(Y0...Yn)/dvj,
        fill the sparse Jacobian:
        |           |
        |  dRi/dvj  |
        |           |

        :param residuals: the residuals (R)
        :param variables: the differentiation variables
        :param n_residuals: number of residuals
        :param n_variables: number of variables
        """
        dres_dvar = lil_matrix((n_residuals, n_variables))
        # end of SoSTrades modif

        out_i = 0
        # Row blocks
        for residual in residuals:
            residual_size = self.sizes[residual]
            # Find the associated discipline
            discipline = self.disciplines[residual]
            residual_jac = discipline.jac[residual]
            # Column blocks
            out_j = 0
            for variable in variables:
                variable_size = self.sizes[variable]
                if residual == variable:
                    # residual Yi-Yi: put -I in the Jacobian
                    ones_mat = (ones(variable_size), 0)
                    shape = (variable_size, variable_size)
                    diag_mat = -dia_matrix(ones_mat, shape=shape)

                    if self.coupling_structure.is_self_coupled(discipline):
                        jac = residual_jac.get(variable, None)
                        if jac is not None:
                            diag_mat += jac
                    dres_dvar[
                    out_i: out_i + variable_size, out_j: out_j + variable_size
                    ] = diag_mat

                else:
                    # block Jacobian
                    jac = residual_jac.get(variable, None)
                    if jac is not None:
                        n_i, n_j = jac.shape
                        assert n_i == residual_size
                        assert n_j == variable_size
                        # Fill the sparse Jacobian block
                        dres_dvar[out_i: out_i + n_i, out_j: out_j + n_j] = jac
                # Shift the column by block width
                out_j += variable_size
            # Shift the row by block height
            out_i += residual_size
        return dres_dvar.real

    def dres_dvar(
            self,
            residuals,
            variables,
            n_residuals,
            n_variables,
            matrix_type=JacobianAssembly.SPARSE,
            transpose=False,
    ):
        """Forms the matrix of partial derivatives of residuals
        Given disciplinary Jacobians dYi(Y0...Yn)/dvj,
        fill the sparse Jacobian:
        |           |
        |  dRi/dvj  |
        |           | (Default value = False)

        :param residuals: the residuals (R)
        :param variables: the differentiation variables
        :param n_residuals: number of residuals
        :param n_variables: number of variables
        :param matrix_type: type of the matrix (Default value = SPARSE)
        :param transpose: if True, transpose the matrix
        """
        if matrix_type == JacobianAssembly.SPARSE:
            if getenv("USE_PETSC", "").lower() in ("true", "1"):
                sparse_dres_dvar = self._dres_dvar_sparse_lil(
                    residuals, variables, n_residuals, n_variables
                )
            else:
                sparse_dres_dvar = self._dres_dvar_sparse(
                    residuals, variables, n_residuals, n_variables
                )
            if transpose:
                return sparse_dres_dvar.T
            return sparse_dres_dvar

        if matrix_type == JacobianAssembly.LINEAR_OPERATOR:
            if transpose:
                return self._dres_dvar_t_linop(
                    residuals, variables, n_residuals, n_variables
                )
            return self._dres_dvar_linop(residuals, variables, n_residuals, n_variables)

        # SoSTrades modif
        if matrix_type == 'func_python':
            return self._dres_dvar_func(residuals, variables,
                                        n_residuals, n_variables)
        # end of SoSTrades modif

        raise TypeError("cannot handle the matrix type")

    # SoSTrades modif
    def _dres_dvar_func(self, residuals, variables, n_residuals,
                        n_variables):
        """Forms the linear operator of partial derivatives of residuals

        :param residuals: the residuals (R)
        :param variables: the differentiation variables
        :param n_residuals: number of residuals
        :param n_variables: number of variables
        """

        # define the linear function
        def dres_dvar(x_array):
            """The linear operator that represents the square matrix dR/dy

            :param x_array: vector multiplied by the matrix
            """
            assert x_array.shape[0] == n_variables
            # initialize the result
            result = zeros(n_residuals)

            out_i = 0
            # Row blocks
            for residual in residuals:
                residual_size = self.sizes[residual]
                # Find the associated discipline
                discipline = self._disciplines[residual]
                residual_jac = discipline.jac[residual]
                # Column blocks
                out_j = 0
                for variable in variables:
                    variable_size = self.sizes[variable]
                    if residual == variable:
                        # residual Yi-Yi: (-I).x = -x
                        sub_x = x_array[out_j:out_j + variable_size]
                        result[out_i:out_i + residual_size] -= sub_x
                    else:
                        # block Jacobian
                        jac = residual_jac.get(variable, None)
                        if jac is not None:
                            sub_x = x_array[out_j:out_j + variable_size]
                            sub_result = jac.dot(sub_x)
                            result[out_i:out_i + residual_size] += sub_result
                    # Shift the column by block width
                    out_j += variable_size
                # Shift the row by block height
                out_i += residual_size
            return result

        return dres_dvar

    # end of SoSTrades modif

    def total_derivatives(
            self,
            in_data,
            functions,
            variables,
            couplings,
            linear_solver="LGMRES",
            mode=JacobianAssembly.AUTO_MODE,
            matrix_type=JacobianAssembly.SPARSE,
            use_lu_fact=False,
            exec_cache_tol=None,
            force_no_exec=False,
            **linear_solver_options
    ):
        """Computes the Jacobian of total derivatives of the coupled system formed by
        the disciplines.

        :param in_data: input data dict
        :param functions: the functions to differentiate
        :param variables: the differentiation variables
        :param couplings: the coupling variables
        :param linear_solver: name of the linear solver
            (Default value = 'lgmres')
        :param mode: linearization mode (auto, direct or adjoint)
            (Default value = AUTO_MODE)
        :param matrix_type: representation of the matrix dR/dy (sparse or
            linear operator) (Default value = SPARSE)
        :param use_lu_fact: if True, factorize dres_dy once
            (Default value = False), unsupported for linear operator mode
        :param force_no_exec: if True, the discipline is not
            re executed, cache is loaded anyway
        :param kwargs: dict of optional parameters
        :returns: the dictionary of dictionary of coupled (total) derivatives
        """
        if not functions:
            return defaultdict(default_dict_factory)

        self.__check_inputs(functions, variables, couplings,
                            matrix_type, use_lu_fact)

        # linearize all the disciplines
        self._add_differentiated_inouts(functions, variables, couplings)

        for disc in self.coupling_structure.disciplines:
            if disc.cache is not None and exec_cache_tol is not None:
                disc.cache_tol = exec_cache_tol
            disc.linearize(in_data, force_no_exec=force_no_exec)

        # compute the sizes from the Jacobians
        self.compute_sizes(functions, variables, couplings)
        n_variables = self.compute_dimension(variables)
        n_functions = self.compute_dimension(functions)
        n_couplings = self.compute_dimension(couplings)

        # SoSTrades modif
        if n_couplings == 0:
            raise ValueError(
                "No couplings detected, cannot solve direct or adjoint system !"
            )
        # end of SoSTrades modif

        # compute the partial derivatives of the residuals
        dres_dx = self.dres_dvar(couplings, variables,
                                 n_couplings, n_variables)

        # compute the partial derivatives of the interest functions
        (dfun_dx, dfun_dy) = ({}, {})
        for fun in functions:
            dfun_dx[fun] = self.dfun_dvar(fun, variables, n_variables)
            dfun_dy[fun] = self.dfun_dvar(fun, couplings, n_couplings)

        mode = self._check_mode(mode, n_variables, n_functions)

        # compute the total derivatives
        if mode == JacobianAssembly.DIRECT_MODE:
            # sparse square matrix dR/dy
            dres_dy = self.dres_dvar(
                couplings, couplings, n_couplings, n_couplings, matrix_type=matrix_type
            )
            # compute the coupled derivatives
            total_derivatives = self.coupled_system.direct_mode(
                functions,
                n_variables,
                n_couplings,
                dres_dx,
                dres_dy,
                dfun_dx,
                dfun_dy,
                linear_solver,
                use_lu_fact=use_lu_fact,
                **linear_solver_options
            )
        elif mode == JacobianAssembly.ADJOINT_MODE:
            # transposed square matrix dR/dy^T
            dres_dy_t = self.dres_dvar(
                couplings,
                couplings,
                n_couplings,
                n_couplings,
                matrix_type=matrix_type,
                transpose=True,
            )
            # compute the coupled derivatives
            total_derivatives = self.coupled_system.adjoint_mode(
                functions,
                dres_dx,
                dres_dy_t,
                dfun_dx,
                dfun_dy,
                linear_solver,
                use_lu_fact=use_lu_fact,
                **linear_solver_options
            )
        else:
            raise ValueError("Incorrect linearization mode " + str(mode))

        return self.split_jac(total_derivatives, variables)

    def _add_differentiated_inouts(self, functions, variables, couplings):
        """Adds functions to the list of differentiated outputs of all disciplines wrt
        couplings, and variables of the discipline.

        :param functions: the functions to differentiate
        :param variables: the differentiation variables
        :param couplings: the coupling variables
        """
        couplings_and_functions = set(couplings) | set(functions)
        couplings_and_variables = set(couplings) | set(variables)

        for discipline in self.coupling_structure.disciplines:
            # outputs
            disc_outputs = discipline.get_output_data_names()
            outputs = list(couplings_and_functions & set(disc_outputs))

            # inputs
            disc_inputs = discipline.get_input_data_names()
            inputs = list(set(disc_inputs) & couplings_and_variables)

            if inputs and outputs:
                discipline.add_differentiated_inputs(inputs)
                discipline.add_differentiated_outputs(outputs)

            # SoSTrades modif
            # If no inputs are couplings but the output is in the discipline we need the jacobian anyway for adjoint method
            # Tocompute sizes of the jac of the output for example
            elif outputs and not inputs:
                discipline.add_differentiated_outputs(outputs)
                disc_inputs = []
                for input_name in discipline.get_input_data_names():
                    try:
                        len(discipline.get_inputs_by_name(input_name))
                        disc_inputs += [input_name, ]
                    except:
                        pass
                discipline.add_differentiated_inputs(list(set(disc_inputs)))

            # - unconsistent check in case of a discipline that has no strong couplings (e.g, a discipline dead-end)

    #             if outputs and not inputs:
    #                 base_msg = (
    #                     "Discipline '{}' has the outputs '{}' that must be "
    #                     "differenciated, but no coupling or design "
    #                     "variables as inputs"
    #                 )
    #                 raise ValueError(base_msg.format(discipline.name, outputs))

    # end of SoSTrades modif

    # Newton step computation
    def compute_newton_step(
            self,
            in_data,
            couplings,
            relax_factor,
            linear_solver="LGMRES",
            matrix_type=JacobianAssembly.LINEAR_OPERATOR,
            **linear_solver_options
    ):
        """Compute Newton step for the the coupled system of residuals formed by the
        disciplines.

        :param in_data: input data dict
        :param couplings: the coupling variables
        :param relax_factor: the relaxation factor
        :param linear_solver: the name of the linear solver
            (Default value = 'lgmres')
        :param matrix_type: representation of the matrix dR/dy (sparse or
            linear operator) (Default value = LINEAR_OPERATOR)
        :param kwargs: optional parameters for the linear solver
        :returns: The Newton step -[dR/dy]^-1 . R as a dict of steps
            per coupling variable
        """
        # linearize the disciplines
        self._add_differentiated_inouts(couplings, couplings, couplings)

        if self.n_processes > 1:
            self.parallel_linearize.configure_linearize_options(
                exec_before_linearize=False)
        # exec_before_linearize is set to False, if you want to come back to old NewtonRaphson
        # put the flag to True
        self.linearize_all_disciplines(in_data, exec_before_linearize=False)

        self.compute_sizes(couplings, couplings, couplings)
        n_couplings = self.compute_dimension(couplings)

        # SoSTrades modif
        # Petsc needs sparse matrix to configure
        if linear_solver.endswith('_PETSC'):
            matrix_type = self.SPARSE
        else:
            matrix_type = self.LINEAR_OPERATOR
        # end of SoSTrades modif

        # compute the partial derivatives of the residuals
        dres_dy = self.dres_dvar(
            couplings, couplings, n_couplings, n_couplings, matrix_type=matrix_type
        )
        # form the residuals
        # convert into array to compute residuals
        in_data = convert_new_type_into_array(
            in_data, self.coupling_structure.disciplines[0].reduced_dm)
        res = self.residuals(in_data, couplings)
        # solve the linear system
        factory = LinearSolversFactory()
        linear_problem = LinearProblem(dres_dy, -relax_factor * res)
        factory.execute(linear_problem, linear_solver, **linear_solver_options)
        newton_step = linear_problem.solution
        self.n_newton_linear_resolutions += 1

        # split the array of steps
        newton_step_dict = {}
        component = 0
        for coupling in couplings:
            size = self.sizes[coupling]
            newton_step_dict[coupling] = newton_step[component: component + size]
            component += size
        return newton_step_dict

    # Newton step computation
    def compute_newton_step_pure(
            self,
            res,
            couplings,
            relax_factor,
            linear_solver="LGMRES",
            matrix_type=JacobianAssembly.LINEAR_OPERATOR,
            **linear_solver_options
    ):
        """Compute Newton step dictionary and let the solver decide how to apply the newton step.
        :param res: residuals for the newton step
        :param couplings: the coupling variables
        :param relax_factor: the relaxation factor
        :param linear_solver: the name of the linear solver
            (Default value = 'lgmres')
        :param matrix_type: representation of the matrix dR/dy (sparse or
            linear operator) (Default value = LINEAR_OPERATOR)
        :param kwargs: optional parameters for the linear solver
        :returns: The Newton step -[dR/dy]^-1 . R as a dict of steps
            per coupling variable
        """

        self.compute_sizes(couplings, couplings, couplings)
        n_couplings = self.compute_dimension(couplings)

        # Petsc needs sparse matrix to configure
        if linear_solver.endswith('_PETSC'):
            matrix_type = self.SPARSE

        # compute the partial derivatives of the residuals
        dres_dy = self.dres_dvar(
            couplings, couplings, n_couplings, n_couplings, matrix_type=matrix_type
        )

        # solve the linear system
        factory = LinearSolversFactory()
        linear_problem = LinearProblem(dres_dy, res)
        factory.execute(linear_problem, linear_solver, **linear_solver_options)
        newton_step = linear_problem.solution
        self.n_newton_linear_resolutions += 1

        # split the array of steps
        newton_step_dict = {}
        component = 0
        for coupling in couplings:
            size = self.sizes[coupling]
            newton_step_dict[coupling] = -relax_factor * \
                                         newton_step[component: component + size]
            component += size

        return newton_step_dict

    def _adjoint_mode(
            self, functions, dres_dx, dres_dy_t, dfun_dx, dfun_dy, linear_solver, **kwargs
    ):
        """Computation of total derivative Jacobian in adjoint mode.

        :param functions: functions to differentiate
        :param dres_dx: Jacobian of residuals wrt design variables
        :param dres_dy: Jacobian of residuals wrt coupling variables
        :param dfun_dx: Jacobian of functions wrt design variables
        :param dfun_dy: Jacobian of functions wrt coupling variables
        :param linear_solver: name of the linear solver
        :param kwargs: optional parameters
        :type kwargs: dict
        :param dres_dy_t: derivatives of the residuals wrt coupling vars
        """
        jac = {}
        # adjoint vector for each interest function
        self.linear_solver.outer_v = []
        # SoSTtrades modif
        parallel = False
        # parallel adjoint for testing purpose
        if parallel:
            solver = deepcopy(self.linear_solver)
            dfun_list = [(f, dfun_dx.copy(), dfun_dy.copy(), dres_dx.copy(),
                          dres_dy_t.copy(), solver, linear_solver) for f in functions]
            with Pool(processes=4) as pool:
                jac_list = pool.map(comp_jac, dfun_list)

            for j, fun in zip(jac_list, functions):
                jac[fun] = j
                dfunction_dy = dfun_dy[fun]
                for _ in range(dfunction_dy.shape[0]):
                    self.n_linear_resolutions += 1
        # end of SoSTtrades modif
        else:
            for fun in functions:
                dfunction_dx = dfun_dx[fun]
                dfunction_dy = dfun_dy[fun]
                jac[fun] = empty(dfunction_dx.shape)
                # compute adjoint vector for each component of the function
                for fun_component in range(dfunction_dy.shape[0]):
                    adjoint = self.linear_solver.solve(
                        dres_dy_t,
                        -dfunction_dy[fun_component, :].T,
                        linear_solver=linear_solver,
                        **kwargs
                    )
                    self.n_linear_resolutions += 1
                    jac[fun][fun_component, :] = (
                            dfunction_dx[fun_component, :] +
                            (dres_dx.T.dot(adjoint)).T
                    )
        return jac

    def __check_inputs(self, functions, variables, couplings, matrix_type, use_lu_fact):
        """Check the inputs before differentiation.

        Args:
            functions: The functions to differentiate.
            variables: The differentiation variables.
            couplings: The coupling variables.
            matrix_type: The type of matrix for linearization.
            use_lu_fact: Whether to use the LU factorization once for all second members.

        Raises:
            ValueError: When the inputs are inconsistent.
        """
        unknown_dvars = set(variables)
        unknown_outs = set(functions)

        for discipline in self.coupling_structure.disciplines:
            inputs = set(discipline.get_input_data_names())
            outputs = set(discipline.get_output_data_names())
            unknown_outs = unknown_outs - outputs
            unknown_dvars = unknown_dvars - inputs

        if unknown_dvars:
            raise ValueError(
                "Some of the specified variables are not "
                + "inputs of the disciplines: "
                + str(unknown_dvars)
                + " possible inputs are: "
                + str(
                    [
                        disc.get_input_data_names()
                        for disc in self.coupling_structure.disciplines
                    ]
                )
            )

        if unknown_outs:
            raise ValueError(
                "Some outputs are not computed by the disciplines:"
                + str(unknown_outs)
                + " available outputs are: "
                + str(
                    [
                        disc.get_output_data_names()
                        for disc in self.coupling_structure.disciplines
                    ]
                )
            )

        for coupling in set(couplings) & set(variables):
            raise ValueError(
                "Variable "
                + str(coupling)
                + " is both a coupling and a design variable"
            )

        if matrix_type not in self.AVAILABLE_MAT_TYPES:
            raise ValueError(
                "Unknown matrix type "
                + str(matrix_type)
                + ", available ones are "
                + str(self.AVAILABLE_MAT_TYPES)
            )

        if use_lu_fact and matrix_type == self.LINEAR_OPERATOR:
            raise ValueError(
                "Unsupported LU factorization for "
                + "LinearOperators! Please use Sparse matrices"
                + " instead"
            )

    def linearize_all_disciplines(
            self,
            input_local_data,  # type: Mapping[str,ndarray]
            force_no_exec=False,
            exec_before_linearize=True
    ):  # type: (...) -> None
        """Linearize all the disciplines.
        Args:
            input_local_data: The input data of the disciplines.
        """
        parallel_linearization_is_working = True

        if self.n_processes > 1 and parallel_linearization_is_working:

            n_disc = len(self.coupling_structure.disciplines)

            inputs_copy_list = [deepcopy(input_local_data)
                                for _ in range(n_disc)]
            self.parallel_linearize.execute(inputs_copy_list)
        else:
            for disc in self.coupling_structure.disciplines:
                disc.linearize(input_local_data, force_no_exec=force_no_exec,
                               exec_before_linearize=exec_before_linearize)


def comp_jac(tup):
    fun, dfun_dx, dfun_dy, dres_dx, dres_dy_t, solver, linear_solver = tup
    dfunction_dx = dfun_dx[fun]
    dfunction_dy = dfun_dy[fun]
    _jac = empty(dfunction_dx.shape)
    # compute adjoint vector for each component of the function
    for fun_component in range(dfunction_dy.shape[0]):
        adjoint = solver.solve(
            dres_dy_t, -dfunction_dy[fun_component, :].T,
            linear_solver=linear_solver)
        _jac[fun_component, :] = dfunction_dx[
                                 fun_component, :] + (dres_dx.T.dot(adjoint)).T
    return _jac
