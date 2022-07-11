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
'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import sys
import platform
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.execution_engine.sos_coupling import get_available_linear_solvers

if platform.system() != 'Windows':
    import petsc4py  # pylint: disable-msg=E0401
    from gemseo.algos.linear_solvers.ksp_lib import _convert_ndarray_to_mat_or_vec  # pylint: disable-msg=E0401
    from sos_trades_core.execution_engine.gemseo_addon.linear_solvers.ksp_lib import PetscKSPAlgos as ksp_lib_petsc
    from petsc4py import PETSc  # pylint: disable-msg=E0401

from scipy.sparse import load_npz
from os.path import dirname, join
import unittest
import pandas as pd
import numpy as np


class TestPetsc(unittest.TestCase):
    """
    Tests on petsc ksp library
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.study_name = 'optim'
        self.ns = f'{self.study_name}'
        self.c_name = "SellarCoupling"
        self.sc_name = "SellarOptimScenario"
        self.repo = 'sostrades_core.sos_processes.test'
        self.proc_name = 'test_sellar_coupling'
        dspace_dict = {'variable': ['x', 'z', 'y_1', 'y_2'],
                       'value': [1., [5., 2.], 1., 1.],
                       'lower_bnd': [0., [-10., 0.], -100., -100.],
                       'upper_bnd': [10., [10., 10.], 100., 100.],
                       'enable_variable': [True, True, True, True],
                       'activated_elem': [[True], [True, True], [True], [True]]}
        self.dspace = pd.DataFrame(dspace_dict)

    def test_01_base_petsc(self):
        if platform.system() != 'Windows':
            comm = PETSc.COMM_WORLD
            size = comm.getSize()
            rank = comm.getRank()

            OptDB = PETSc.Options()
            m = OptDB.getInt('m', 8)
            n = OptDB.getInt('n', 7)

            ''' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Compute the matrix and right-hand-side vector that define
                    the linear system, Ax = b.
                - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - '''
            '''
                Create parallel matrix, specifying only its global dimensions.
                When using MatCreate(), the matrix format can be specified at
                runtime. Also, the parallel partitioning of the matrix is
                determined by PETSc at runtime.
            
                Performance tuning note:  For problems of substantial size,
                preallocation of matrix memory is crucial for attaining good
                performance. See the matrix chapter of the users manual for details.
            '''

            A = PETSc.Mat().create(comm=comm)
            A.setSizes((m * n, m * n))
            A.setFromOptions()
            A.setPreallocationNNZ((5, 5))

            '''
                Currently, all PETSc parallel matrix formats are partitioned by
                contiguous chunks of rows across the processors.  Determine which
                rows of the matrix are locally owned.
            '''
            Istart, Iend = A.getOwnershipRange()

            '''
                Set matrix elements for the 2-D, five-point stencil in parallel.
                - Each processor needs to insert only elements that it owns
                locally (but any non-local elements will be sent to the
                appropriate processor during matrix assembly).
                - Always specify global rows and columns of matrix entries.
            
                Note: this uses the less common natural ordering that orders first
                all the unknowns for x = h then for x = 2h etc; Hence you see J = Ii +- n
                instead of J = I +- m as you might expect. The more standard ordering
                would first do all variables for y = h, then y = 2h etc.
            '''

            for Ii in range(Istart, Iend):
                v = -1.0
                i = int(Ii / n)
                j = int(Ii - i * n)

                if (i > 0):
                    J = Ii - n
                    A.setValues(Ii, J, v, addv=True)
                if (i < m - 1):
                    J = Ii + n
                    A.setValues(Ii, J, v, addv=True)
                if (j > 0):
                    J = Ii - 1
                    A.setValues(Ii, J, v, addv=True)
                if (j < n - 1):
                    J = Ii + 1
                    A.setValues(Ii, J, v, addv=True)

                v = 4.0
                A.setValues(Ii, Ii, v, addv=True)

            '''
                Assemble matrix, using the 2-step process:
                MatAssemblyBegin(), MatAssemblyEnd()
                Computations can be done while messages are in transition
                by placing code between these two statements.
            '''

            A.assemblyBegin(A.AssemblyType.FINAL)
            A.assemblyEnd(A.AssemblyType.FINAL)
            ''' A is symmetric. Set symmetric flag to enable ICC/Cholesky preconditioner '''

            A.setOption(A.Option.SYMMETRIC, True)

            '''
                Create parallel vectors.
                - We form 1 vector from scratch and then duplicate as needed.
                - When using VecCreate(), VecSetSizes and VecSetFromOptions()
                in this example, we specify only the
                vector's global dimension; the parallel partitioning is determined
                at runtime.
                - When solving a linear system, the vectors and matrices MUST
                be partitioned accordingly.  PETSc automatically generates
                appropriately partitioned matrices and vectors when MatCreate()
                and VecCreate() are used with the same communicator.
                - The user can alternatively specify the local vector and matrix
                dimensions when more sophisticated partitioning is needed
                (replacing the PETSC_DECIDE argument in the VecSetSizes() statement
                below).
            '''

            u = PETSc.Vec().create(comm=comm)
            u.setSizes(m * n)
            u.setFromOptions()

            b = u.duplicate()
            x = b.duplicate()

            '''
                Set exact solution; then compute right-hand-side vector.
                By default we use an exact solution of a vector with all
                elements of 1.0;  
            '''
            u.set(1.0)
            b = A(u)

            '''
                View the exact solution vector if desired
            '''
            flg = OptDB.getBool('view_exact_sol', False)
            if flg:
                u.view()

            ''' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Create the linear solver and set various options
                - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - '''
            ksp = PETSc.KSP().create(comm=comm)

            '''
                Set operators. Here the matrix that defines the linear system
                also serves as the preconditioning matrix.
            '''
            ksp.setOperators(A, A)

            '''
                Set linear solver defaults for this problem (optional).
                - By extracting the KSP and PC contexts from the KSP context,
                we can then directly call any KSP and PC routines to set
                various options.
                - The following two statements are optional; all of these
                parameters could alternatively be specified at runtime via
                KSPSetFromOptions().  All of these defaults can be
                overridden at runtime, as indicated below.
            '''
            rtol = 1.e-2 / ((m + 1) * (n + 1))
            ksp.setTolerances(rtol=rtol, atol=1.e-50)

            '''
            Set runtime options, e.g.,
                -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
            These options will override those specified above as long as
            KSPSetFromOptions() is called _after_ any other customization
            routines.
            '''
            ksp.setFromOptions()

            ''' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                Solve the linear system
                - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - '''

            ksp.solve(b, x)

            ''' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                Check the solution and clean up
                - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - '''
            x = x - u  # x.axpy(-1.0,u)
            norm = x.norm(PETSc.NormType.NORM_2)
            its = ksp.getIterationNumber()

            '''
                Print convergence information.  PetscPrintf() produces a single
                print statement from all processes that share a communicator.
                An alternative is PetscFPrintf(), which prints to a file.
            '''
            if norm > rtol * 10:
                PETSc.Sys.Print(
                    'Norm of error {}, Iterations {}'.format(norm, its), comm=comm)
            else:
                if size == 1:
                    PETSc.Sys.Print('- Serial OK', comm=comm)
                else:
                    PETSc.Sys.Print('- Parallel OK', comm=comm)

            self.assertLessEqual(norm, rtol * 10)

    def test_02_petsc_on_witness_matrix(self):
        if platform.system() != 'Windows':
            comm = PETSc.COMM_WORLD
            size = comm.getSize()
            rank = comm.getRank()
            # rnorm < MAX (rtol * rnorm_0, abstol)
            options = {'solver_type': 'gmres', 'tol': 1.0e-50,
                       'atol': 1.0e-10, 'dtol': 1.0e5, 'max_iter': 100000}
            ksp = PETSc.KSP().create()
            ksp.setType(options["solver_type"])
            ksp.setTolerances(options["tol"], options["atol"],
                              options["dtol"], options["max_iter"])
            ksp.setConvergenceHistory()
            lhs = load_npz(join(dirname(__file__), 'data', 'a_mat_sparse.npz'))
            a_mat = _convert_ndarray_to_mat_or_vec(lhs)
            ksp.setOperators(a_mat)
            prec_type = 'gasm'  # ilu
            if prec_type is not None:
                pc = ksp.getPC()
                pc.setType(prec_type)
                pc.setUp()
            rhs = pd.read_pickle(join(dirname(__file__), 'data', 'b_vec.pkl'))
            b_mat = _convert_ndarray_to_mat_or_vec(rhs)
            solution = b_mat.duplicate()

            ksp.solve(b_mat, solution)
            solution_final = solution.getArray()
            convergence_info = ksp.reason
            print(convergence_info)

            self.assertGreater(convergence_info, 0)

    def test_03_Sellar_with_petsc(self):
        if platform.system() != 'Windows':

            exec_eng = ExecutionEngine(self.study_name)
            factory = exec_eng.factory

            builder = factory.get_builder_from_process(repo=self.repo,
                                                       mod_id='test_sellar_coupling')

            exec_eng.factory.set_builders_to_coupling_builder(builder)

            exec_eng.configure()

            # Sellar inputs
            local_dv = 10.
            values_dict = {}
            values_dict[f'{self.ns}.{self.c_name}.chain_linearize'] = True
            values_dict[f'{self.ns}.{self.c_name}.sub_mda_class'] = "MDANewtonRaphson"
            values_dict[f'{self.ns}.{self.c_name}.linear_solver_MDA'] = "GMRES_PETSC"
            values_dict[f'{self.ns}.{self.c_name}.linear_solver_MDA_preconditioner'] = "gasm"
            values_dict[f'{self.ns}.{self.c_name}.x'] = np.array([1.])
            values_dict[f'{self.ns}.{self.c_name}.y_1'] = np.array([1.])
            values_dict[f'{self.ns}.{self.c_name}.y_2'] = np.array([1.])
            values_dict[f'{self.ns}.{self.c_name}.z'] = np.array([1., 1.])
            values_dict[f'{self.ns}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
            exec_eng.load_study_from_input_dict(values_dict)

            mda = exec_eng.root_process.sos_disciplines[0]

            sub_mda_class = mda.sub_mda_list[0]

            self.assertEqual(sub_mda_class.linear_solver_options['preconditioner_type'], 'gasm'
                             )
            self.assertEqual(
                sub_mda_class.linear_solver_options['solver_type'], 'gmres')

            exec_eng.execute()

            obj, y_1, y_2 = exec_eng.root_process.sos_disciplines[0].get_sosdisc_outputs([
                                                                                         "obj", "y_1", "y_2"])

            obj_ref = np.array([14.32662157])
            y_1_ref = np.array([2.29689011])
            y_2_ref = np.array([3.51554944])

            self.assertAlmostEqual(obj_ref, obj.real, delta=1e-5)
            self.assertAlmostEqual(y_1_ref, y_1.real, delta=1e-5)
            self.assertAlmostEqual(y_2_ref, y_2.real, delta=1e-5)

    def test_04_Sellar_available_linear_solver_petsc_and_preconditioner(self):

        if platform.system() != 'Windows':

            available_linear_solver = ksp_lib_petsc.AVAILABLE_LINEAR_SOLVER
            available_preconditioner = ksp_lib_petsc.AVAILABLE_PRECONDITIONER

            for linear_solver_MDA in available_linear_solver:
                for preconditioner in available_preconditioner:

                    print(
                        f'--> Run with linear solver MDA: {linear_solver_MDA} and preconditioner: {preconditioner}')

                    exec_eng = ExecutionEngine(self.study_name)
                    factory = exec_eng.factory

                    builder = factory.get_builder_from_process(repo=self.repo,
                                                               mod_id='test_sellar_coupling')

                    exec_eng.factory.set_builders_to_coupling_builder(builder)

                    exec_eng.configure()

                    # Sellar inputs
                    local_dv = 10.
                    values_dict = {}
                    values_dict[f'{self.ns}.{self.c_name}.chain_linearize'] = True
                    values_dict[f'{self.ns}.{self.c_name}.sub_mda_class'] = "MDANewtonRaphson"
                    values_dict[f'{self.ns}.{self.c_name}.linear_solver_MDA'] = linear_solver_MDA
                    values_dict[f'{self.ns}.{self.c_name}.linear_solver_MDA_preconditioner'] = preconditioner
                    values_dict[f'{self.ns}.{self.c_name}.x'] = np.array([1.])
                    values_dict[f'{self.ns}.{self.c_name}.y_1'] = np.array([1.])
                    values_dict[f'{self.ns}.{self.c_name}.y_2'] = np.array([1.])
                    values_dict[f'{self.ns}.{self.c_name}.z'] = np.array([
                                                                         1., 1.])
                    values_dict[f'{self.ns}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
                    exec_eng.load_study_from_input_dict(values_dict)

                    mda = exec_eng.root_process.sos_disciplines[0]

                    sub_mda_class = mda.sub_mda_list[0]

                    self.assertEqual(sub_mda_class.linear_solver_options['preconditioner_type'], preconditioner
                                     )
                    self.assertEqual(
                        sub_mda_class.linear_solver_options['solver_type'], linear_solver_MDA.split('_PETSC')[0].lower())

                    exec_eng.execute()

                    obj, y_1, y_2 = exec_eng.root_process.sos_disciplines[0].get_sosdisc_outputs([
                                                                                                 "obj", "y_1", "y_2"])

                    obj_ref = np.array([14.32662157])
                    y_1_ref = np.array([2.29689011])
                    y_2_ref = np.array([3.51554944])

                    self.assertAlmostEqual(obj_ref, obj.real, delta=1e-5)
                    self.assertAlmostEqual(y_1_ref, y_1.real, delta=1e-5)
                    self.assertAlmostEqual(y_2_ref, y_2.real, delta=1e-5)


if '__main__' == __name__:

    cls = TestPetsc()
    cls.setUp()
    cls.test_04_Sellar_available_linear_solver_petsc_and_preconditioner()
