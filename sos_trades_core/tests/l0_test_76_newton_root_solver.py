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
import unittest
import numpy as np

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.sos_wrapping.test_discs.polynom_disc import Polynom


class TestNewtonRootSolver(unittest.TestCase):
    """
    Newton Root Solver test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.dirs_to_del = []
        self.namespace = 'MyCase'
        self.study_name = f'{self.namespace}'
        self.exec_eng = ExecutionEngine(self.namespace)
        self.factory = self.exec_eng.factory
        self.exec_eng.ns_manager.add_ns_def({'ns_z': self.study_name,
                                             'ns_float': 'toto'})

    def test_01_associate_residual_builders_to_nr_solver(self):

        builder = self.exec_eng.factory.create_builder_newton_root_solver(
            'NewtonSolver')

        builder.set_builder_info(
            'residual_builders', self.exec_eng.factory.get_builder_from_class_name('SolveProblem', 'Polynom', [
                'sos_trades_core.sos_wrapping.test_discs']))

        builder.set_builder_info(
            'residual_infos', {'residual_variable': 'z',
                               'unknown_variable': 'newton_unknowns',
                               'residual_ns_name': 'ns_z'})

        disc = builder.build()

        # Only one sub discipline under the newtons olver
        self.assertEqual(len(disc.sos_disciplines), 1)
        self.assertEqual(Polynom, disc.sos_disciplines[0].__class__)

    def test_02_configure_nr_solver_with_wrong_ns(self):

        builder = self.exec_eng.factory.create_builder_newton_root_solver(
            'NewtonSolver')

        builder.set_builder_info(
            'residual_builders', self.exec_eng.factory.get_builder_from_class_name('SolveProblem', 'Polynom', [
                'sos_trades_core.sos_wrapping.test_discs']))

        builder.set_builder_info(
            'residual_infos', {'residual_variable': 'z',
                               'unknown_variable': 'newton_unknowns',
                               'residual_ns_name': 'ns_wrong'})

        self.exec_eng.factory.set_builders_to_coupling_builder(builder)
        with self.assertRaises(Exception) as cm:
            self.exec_eng.configure()

        error_message = 'The namespace ns_wrong has not been declared in residual builders, the unknown variable cannot be found with the given unknown_ns_name'
        self.assertTrue(str(cm.exception) == error_message)

    def test_03_configure_nr_solver_with_wrong_ns_for_unknown(self):

        builder = self.exec_eng.factory.create_builder_newton_root_solver(
            'NewtonSolver')

        builder.set_builder_info(
            'residual_builders', self.exec_eng.factory.get_builder_from_class_name('SolveProblem', 'Polynom', [
                'sos_trades_core.sos_wrapping.test_discs']))

        builder.set_builder_info(
            'residual_infos', {'residual_variable': 'z',
                               'unknown_variable': 'float_unknown',
                               'residual_ns_name': 'ns_z'})

        self.exec_eng.factory.set_builders_to_coupling_builder(builder)
        with self.assertRaises(Exception) as cm:
            self.exec_eng.configure()
        print(cm.exception)
        error_message = "The unknown variable MyCase.float_unknown does not exist in the inputs of the given residual builders \nMaybe the namespace is not coherent with existing variable names in residual builders : ['toto.float_unknown']"
        self.assertTrue(str(cm.exception) == error_message)

    def test_04_configure_nr_solver_with_wrong_type_for_unknown(self):

        builder = self.exec_eng.factory.create_builder_newton_root_solver(
            'NewtonSolver')

        builder.set_builder_info(
            'residual_builders', self.exec_eng.factory.get_builder_from_class_name('SolveProblem', 'Polynom', [
                'sos_trades_core.sos_wrapping.test_discs']))

        builder.set_builder_info(
            'residual_infos', {'residual_variable': 'z',
                               'unknown_variable': 'float_unknown',
                               'residual_ns_name': 'ns_z',
                               'unknown_ns_name': 'ns_float'})

        self.exec_eng.factory.set_builders_to_coupling_builder(builder)
        with self.assertRaises(Exception) as cm:
            self.exec_eng.configure()

        error_message = "Newton Root solver only uses arrays : The unknown variable toto.float_unknown must be specified as an array in residual builders for Newton Root Solver"
        self.assertTrue(str(cm.exception) == error_message)

    def test_05_execute_nr_solver(self):

        builder = self.exec_eng.factory.create_builder_newton_root_solver(
            'NewtonSolver')

        builder.set_builder_info(
            'residual_builders', self.exec_eng.factory.get_builder_from_class_name('SolveProblem', 'Polynom', [
                'sos_trades_core.sos_wrapping.test_discs']))

        builder.set_builder_info(
            'residual_infos', {'residual_variable': 'z',
                               'unknown_variable': 'newton_unknowns',
                               'residual_ns_name': 'ns_z'})

        self.exec_eng.factory.set_builders_to_coupling_builder(builder)

        self.exec_eng.configure()

        values_dict = {self.study_name + '.NewtonSolver.x0': np.zeros(2)}
        self.exec_eng.load_study_from_input_dict(values_dict)

        self.exec_eng.execute()

        x_final = self.exec_eng.dm.get_value('MyCase.NewtonSolver.x_final')

        self.assertTrue(np.allclose(
            x_final, [0., 1.]) or np.allclose(x_final, [1., 2.]))

    def test_06_execute_nr_solver_complex_step(self):

        builder = self.exec_eng.factory.create_builder_newton_root_solver(
            'NewtonSolver')

        builder.set_builder_info(
            'residual_builders', self.exec_eng.factory.get_builder_from_class_name('SolveProblem', 'Polynom', [
                'sos_trades_core.sos_wrapping.test_discs']))

        builder.set_builder_info(
            'residual_infos', {'residual_variable': 'z',
                               'unknown_variable': 'newton_unknowns',
                               'residual_ns_name': 'ns_z'})

        self.exec_eng.factory.set_builders_to_coupling_builder(builder)

        self.exec_eng.configure()

        values_dict = {self.study_name + '.NewtonSolver.x0': np.zeros(2),
                       self.study_name + '.NewtonSolver.NR_diff_mode': 'Complex step'}
        self.exec_eng.load_study_from_input_dict(values_dict)

        self.exec_eng.execute()

        x_final = self.exec_eng.dm.get_value('MyCase.NewtonSolver.x_final')

        self.assertTrue(np.allclose(
            x_final, [0., 1.]) or np.allclose(x_final, [1., 2.]))

    def test_06_execute_nr_solver_analytic_jacobian(self):

        builder = self.exec_eng.factory.create_builder_newton_root_solver(
            'NewtonSolver')

        builder.set_builder_info(
            'residual_builders', self.exec_eng.factory.get_builder_from_class_name('SolveProblem', 'Polynom', [
                'sos_trades_core.sos_wrapping.test_discs']))

        builder.set_builder_info(
            'residual_infos', {'residual_variable': 'z',
                               'unknown_variable': 'newton_unknowns',
                               'residual_ns_name': 'ns_z'})

        self.exec_eng.factory.set_builders_to_coupling_builder(builder)

        self.exec_eng.configure()

        values_dict = {self.study_name + '.NewtonSolver.x0': np.zeros(2),
                       self.study_name + '.NewtonSolver.NR_diff_mode': 'Analytic'}
        self.exec_eng.load_study_from_input_dict(values_dict)

        self.exec_eng.execute()

        x_final = self.exec_eng.dm.get_value('MyCase.NewtonSolver.x_final')

        self.assertTrue(np.allclose(
            x_final, [0., 1.]) or np.allclose(x_final, [1., 2.]))


if '__main__' == __name__:
    cls = TestNewtonRootSolver()
    cls.setUp()
    cls.test_06_execute_nr_solver_analytic_jacobian()
