'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/25-2024/05/16 Copyright 2023 Capgemini

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
import unittest
from os import getenv, remove
from os.path import dirname, join
from pathlib import Path
from tempfile import gettempdir
from time import sleep

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.sos_processes.test.test_sellar_opt_discopt.usecase import (
    Study as study_sellar_opt,
)


class TestStructuringInputs(unittest.TestCase):
    """
    Class to test behaviour of structuring inputs during configure step
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.file_to_del = []
        self.namespace = 'MyCase'
        self.study_name = f'{self.namespace}'
        self.repo = 'sostrades_core.sos_processes.test'
        self.base_path = 'sostrades_core.sos_wrapping.test_discs'
        self.exec_eng = ExecutionEngine(self.namespace)
        self.factory = self.exec_eng.factory
        self.root_dir = gettempdir()

    def tearDown(self):
        for file in self.file_to_del:
            if Path(file).is_file():
                remove(file)
                sleep(0.5)

    def test_01_configure_discipline_with_setup_sos_discipline(self):

        self.exec_eng.ns_manager.add_ns(
            'ns_ac', self.exec_eng.study_name)
        builder_process = self.exec_eng.factory.get_builder_from_module(
            'Disc1', 'sostrades_core.sos_wrapping.test_discs.disc1_setup_sos_discipline.Disc1')
        self.exec_eng.factory.set_builders_to_coupling_builder(
            builder_process)

        self.exec_eng.configure()
        self.exec_eng.load_study_from_input_dict({})
        print(self.exec_eng.display_treeview_nodes())
        disc_to_conf = self.exec_eng.root_process.get_disciplines_to_configure()
        self.assertListEqual(disc_to_conf, [])

        full_values_dict = {}
        full_values_dict[self.study_name + '.x'] = 1
        full_values_dict[self.study_name + '.Disc1.a'] = 10
        full_values_dict[self.study_name + '.Disc1.b'] = 3

        self.exec_eng.dm.set_values_from_dict(full_values_dict)
        disc_to_conf = self.exec_eng.root_process.get_disciplines_to_configure()
        self.assertListEqual(disc_to_conf, [])
        disc_1 = self.exec_eng.dm.get_disciplines_with_name('MyCase.Disc1')[0]
        structuring_variables_disc1 = list(disc_1._structuring_variables.keys())
        self.assertListEqual(structuring_variables_disc1, ['AC_list',
                                                           disc_1.LINEARIZATION_MODE,
                                                           disc_1.CACHE_TYPE,
                                                           disc_1.CACHE_FILE_PATH,
                                                           disc_1.DEBUG_MODE, 'dyn_input_2'])

        self.exec_eng.load_study_from_input_dict(full_values_dict)
        print(self.exec_eng.display_treeview_nodes())

        full_values_dict[self.study_name + '.x'] = 2

        self.exec_eng.dm.set_values_from_dict(full_values_dict)
        disc_to_conf = self.exec_eng.root_process.get_disciplines_to_configure()
        self.assertListEqual(disc_to_conf, [])
        disc_1 = self.exec_eng.dm.get_disciplines_with_name('MyCase.Disc1')[0]
        structuring_variables_disc1 = list(disc_1._structuring_variables.keys())
        self.assertListEqual(structuring_variables_disc1, ['AC_list',
                                                           disc_1.LINEARIZATION_MODE,
                                                           disc_1.CACHE_TYPE,
                                                           disc_1.CACHE_FILE_PATH,
                                                           disc_1.DEBUG_MODE,
                                                           'dyn_input_2'])

        self.exec_eng.load_study_from_input_dict(full_values_dict)
        print(self.exec_eng.display_treeview_nodes())

        full_values_dict[self.study_name + '.AC_list'] = ['AC1', 'AC2']

        self.exec_eng.dm.set_values_from_dict(full_values_dict)
        disc_to_conf = self.exec_eng.root_process.get_disciplines_to_configure()
        self.assertListEqual([d.sos_name for d in disc_to_conf], ['Disc1'])

        self.assertTrue(self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Disc1')[0].check_structuring_variables_changes())
        self.exec_eng.load_study_from_input_dict(full_values_dict)
        self.assertFalse(self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Disc1')[0].check_structuring_variables_changes())
        self.assertListEqual(self.exec_eng.dm.get_disciplines_with_name('MyCase.Disc1')[
                                 0]._structuring_variables['AC_list'], ['AC1', 'AC2'])
        disc_to_conf = self.exec_eng.root_process.get_disciplines_to_configure()
        self.assertListEqual(disc_to_conf, [])
        print(self.exec_eng.display_treeview_nodes())

        full_values_dict[self.study_name + '.AC_list'] = ['AC1', 'AC3', 'AC4']

        self.exec_eng.dm.set_values_from_dict(full_values_dict)
        disc_to_conf = self.exec_eng.root_process.get_disciplines_to_configure()
        self.assertListEqual([d.sos_name for d in disc_to_conf], ['Disc1'])

        self.assertTrue(self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Disc1')[0].check_structuring_variables_changes())
        self.exec_eng.load_study_from_input_dict(full_values_dict)
        self.assertFalse(self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Disc1')[0].check_structuring_variables_changes())
        self.assertListEqual(self.exec_eng.dm.get_disciplines_with_name('MyCase.Disc1')[
                                 0]._structuring_variables['AC_list'], ['AC1', 'AC3', 'AC4'])
        disc_to_conf = self.exec_eng.root_process.get_disciplines_to_configure()
        self.assertListEqual(disc_to_conf, [])
        print(self.exec_eng.display_treeview_nodes())

        self.exec_eng.dm.set_values_from_dict(full_values_dict)
        disc_to_conf = self.exec_eng.root_process.get_disciplines_to_configure()
        self.assertListEqual(disc_to_conf, [])

        self.exec_eng.load_study_from_input_dict(full_values_dict)
        print(self.exec_eng.display_treeview_nodes())

        full_values_dict[self.study_name + '.AC_list'] = ['AC3', 'AC4']

        self.exec_eng.dm.set_values_from_dict(full_values_dict)
        disc_to_conf = self.exec_eng.root_process.get_disciplines_to_configure()
        self.assertListEqual([d.sos_name for d in disc_to_conf], ['Disc1'])

        self.assertTrue(self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Disc1')[0].check_structuring_variables_changes())
        self.exec_eng.load_study_from_input_dict(full_values_dict)
        self.assertFalse(self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Disc1')[0].check_structuring_variables_changes())
        self.assertListEqual(self.exec_eng.dm.get_disciplines_with_name('MyCase.Disc1')[
                                 0]._structuring_variables['AC_list'], ['AC3', 'AC4'])

        disc_to_conf = self.exec_eng.root_process.get_disciplines_to_configure()
        self.assertListEqual(disc_to_conf, [])
        print(self.exec_eng.display_treeview_nodes())

        self.exec_eng.execute()


    def _test_05_proxycoupling_numerical_inputs_including_petsc(self):
        """
        Test proper definition of coupling numerical inputs, possible values, etc. and execute using LGMRES with
        GSOrNewtonMDA.

        """
        # TODO: test is deactivated because GSorNewtonMDA needs updating to EEv4 style
        repo_discopt = 'sostrades_core.sos_processes.test'
        proc_name_discopt = 'test_sellar_opt_discopt'
        builder = self.exec_eng.factory.get_builder_from_process(repo=repo_discopt,
                                                                 mod_id=proc_name_discopt)

        self.exec_eng.factory.set_builders_to_coupling_builder(builder)

        self.exec_eng.load_study_from_input_dict({})

        coupling_inputs = {'inner_mda_name': {'type': 'string', 'default': 'MDAJacobi',
                                              'possible_values': ['MDAJacobi', 'MDAGaussSeidel', 'MDANewtonRaphson',
                                                                  'MDAQuasiNewton', 'MDAGSNewton',
                                                                  'GSorNewtonMDA', 'MDASequential']},
                           'max_mda_iter': {'type': 'int', 'default': 30, 'possible_values': None},
                           'n_processes': {'type': 'int', 'default': 1, 'possible_values': None},
                           'chain_linearize': {'type': 'bool', 'default': False, 'possible_values': [True, False]},
                           'tolerance': {'type': 'float', 'default': 1e-06, 'possible_values': None},
                           'use_lu_fact': {'type': 'bool', 'default': False, 'possible_values': [True, False]},
                           'warm_start': {'type': 'bool', 'default': False, 'possible_values': [True, False]},
                           'acceleration_method': {'type': 'string', 'default': 'Alternate2Delta',
                                                   'possible_values': ['Alternate2Delta', 'secant', 'none']},

                           'n_subcouplings_parallel': {'type': 'int', 'default': 1, 'possible_values': None},
                           'tolerance_gs': {'type': 'float', 'default': 10.0, 'possible_values': None},
                           'over_relaxation_factor': {'type': 'float', 'default': 0.99, 'possible_values': None},
                           'epsilon0': {'type': 'float', 'default': 1e-06, 'possible_values': None},
                           'linear_solver_MDO': {'type': 'string'},
                           'linear_solver_MDO_preconditioner': {'type': 'string'},
                           'linear_solver_MDO_options': {'type': 'dict', 'default': {'max_iter': 1000, 'tol': 1e-08},
                                                         'possible_values': None},
                           'linear_solver_MDA': {'type': 'string'},
                           'linear_solver_MDA_preconditioner': {'type': 'string'},
                           'linear_solver_MDA_options': {'type': 'dict', 'default': {'max_iter': 1000, 'tol': 1e-08},
                                                         'possible_values': None},
                           'group_mda_disciplines': {'type': 'bool', 'default': False,
                                                     'possible_values': [True, False]},
                           'propagate_cache_to_children': {'type': 'bool', 'possible_values': [True, False],
                                                           'default': False},
                           # 'authorize_self_coupled_disciplines': {'type': 'bool', 'possible_values': [True, False],
                           #                                        'default': False},
                           'linearization_mode': {'type': 'string', 'default': 'finite_differences',
                                                  'possible_values': ['auto', 'direct', 'adjoint', 'reverse',
                                                                      'finite_differences', 'complex_step']},
                           'cache_type': {'type': 'string', 'default': 'None',
                                          'possible_values': ['None', 'SimpleCache']},
                           'cache_file_path': {'type': 'string', 'default': '', 'possible_values': None},
                           'debug_mode': {'type': 'string', 'default': '',
                                          'possible_values': ["", "nan", "input_change", "linearize_data_change",
                                                              "min_max_couplings", "all"], }
                           }

        if getenv("USE_PETSC", "").lower() not in ("true", "1"):
            coupling_inputs['linear_solver_MDO']['default'] = 'GMRES'
            coupling_inputs['linear_solver_MDA']['default'] = 'GMRES'
            coupling_inputs['linear_solver_MDO']['possible_values'] = [
                'LGMRES', 'GMRES', 'BICG', 'QMR', 'BICGSTAB', 'DEFAULT']
            coupling_inputs['linear_solver_MDA']['possible_values'] = [
                'LGMRES', 'GMRES', 'BICG', 'QMR', 'BICGSTAB', 'DEFAULT']
            coupling_inputs['linear_solver_MDO_preconditioner']['default'] = 'None'
            coupling_inputs['linear_solver_MDA_preconditioner']['default'] = 'None'
            coupling_inputs['linear_solver_MDO_preconditioner']['possible_values'] = [
                'None', 'ilu']
            coupling_inputs['linear_solver_MDA_preconditioner']['possible_values'] = [
                'None', 'ilu']
        else:
            coupling_inputs['linear_solver_MDO']['default'] = 'GMRES_PETSC'
            coupling_inputs['linear_solver_MDA']['default'] = 'GMRES_PETSC'
            coupling_inputs['linear_solver_MDO']['possible_values'] = ['GMRES_PETSC', 'LGMRES_PETSC',
                                                                       'BICG_PETSC', 'BCGS_PETSC', 'LGMRES', 'GMRES',
                                                                       'BICG', 'QMR', 'BICGSTAB', 'DEFAULT']
            coupling_inputs['linear_solver_MDA']['possible_values'] = ['GMRES_PETSC', 'LGMRES_PETSC',
                                                                       'BICG_PETSC', 'BCGS_PETSC', 'LGMRES', 'GMRES',
                                                                       'BICG', 'QMR', 'BICGSTAB', 'DEFAULT']
            coupling_inputs['linear_solver_MDO_preconditioner']['default'] = 'gasm'
            coupling_inputs['linear_solver_MDA_preconditioner']['default'] = 'gasm'
            coupling_inputs['linear_solver_MDO_preconditioner']['possible_values'] = [
                'None', 'jacobi', 'ilu', 'gasm']
            coupling_inputs['linear_solver_MDA_preconditioner']['possible_values'] = [
                'None', 'jacobi', 'ilu', 'gasm']

        # check numerical inputs of root_process coupling
        self.assertListEqual(list(coupling_inputs.keys()), list(
            self.exec_eng.root_process.get_data_in().keys()))

        for input_name, input_dict in coupling_inputs.items():
            for key, value in input_dict.items():
                self.assertEqual(
                    self.exec_eng.root_process.get_data_in()[input_name][key], value)

        self.assertEqual(self.exec_eng.dm.get_data(
            'MyCase.SellarOptimScenario.SellarCoupling.tolerance_gs', 'default'), 10.0)
        self.assertEqual(self.exec_eng.dm.get_data(
            'MyCase.SellarOptimScenario.SellarCoupling.tolerance_gs', 'value'), 10.0)

        uc_cls = study_sellar_opt()
        uc_cls.study_name = 'MyCase'
        dict_values = uc_cls.setup_usecase()
        dict_values[0]['MyCase.SellarOptimScenario.SellarCoupling.inner_mda_name'] = 'GSorNewtonMDA'
        dict_values[0]['MyCase.SellarOptimScenario.SellarCoupling.max_mda_iter'] = 20
        dict_values[0]['MyCase.SellarOptimScenario.SellarCoupling.tolerance'] = 1e-3
        dict_values[0]['MyCase.SellarOptimScenario.SellarCoupling.over_relaxation_factor'] = 0.85
        dict_values[0]['MyCase.SellarOptimScenario.SellarCoupling.epsilon0'] = 1e-4
        dict_values[0]['MyCase.SellarOptimScenario.SellarCoupling.use_lu_fact'] = True
        dict_values[0]['MyCase.SellarOptimScenario.SellarCoupling.linear_solver_MDO'] = 'BICG'
        dict_values[0]['MyCase.SellarOptimScenario.SellarCoupling.linear_solver_MDO_preconditioner'] = 'None'
        dict_values[0]['MyCase.SellarOptimScenario.SellarCoupling.linear_solver_MDO_options'] = {
            'max_iter': 500, 'tol': 1e-07}
        dict_values[0]['MyCase.SellarOptimScenario.SellarCoupling.linear_solver_MDA'] = 'LGMRES'
        dict_values[0]['MyCase.SellarOptimScenario.SellarCoupling.linear_solver_MDA_preconditioner'] = 'ilu'
        dict_values[0]['MyCase.SellarOptimScenario.SellarCoupling.linear_solver_MDA_options'] = {
            'max_iter': 600, 'tol': 1e-10}

        dict_values[0]['MyCase.SellarOptimScenario.formulation'] = 'DisciplinaryOpt'

        self.exec_eng.load_study_from_input_dict(dict_values[0])

        self.assertEqual(self.exec_eng.dm.get_data(
            'MyCase.SellarOptimScenario.SellarCoupling.tolerance_gs', 'default'), 10.0)
        self.assertEqual(self.exec_eng.dm.get_data(
            'MyCase.SellarOptimScenario.SellarCoupling.tolerance_gs', 'value'), 10.0)

        # check numerical inputs of root_process coupling
        self.assertListEqual(list(coupling_inputs.keys()), list(
            self.exec_eng.root_process.get_data_in().keys()))

        for input_name, input_dict in coupling_inputs.items():
            for key, value in input_dict.items():
                self.assertEqual(
                    self.exec_eng.root_process.get_data_in()[input_name][key], value)

        coupling_sellar = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.SellarOptimScenario.SellarCoupling')[0]

        self.exec_eng.prepare_execution()
        self.assertEqual(coupling_sellar.discipline_wrapp.discipline.linear_solver, 'LGMRES')
        self.assertEqual(coupling_sellar.discipline_wrapp.discipline.tolerance, 1e-3)
        self.assertEqual(coupling_sellar.discipline_wrapp.discipline.inner_mdas[0].tolerance, 1e-03)
        self.assertEqual(coupling_sellar.discipline_wrapp.discipline.linear_solver_tolerance, 1e-10)
        self.assertEqual(coupling_sellar.discipline_wrapp.discipline.linear_solver_settings, {
            'max_iter': 600, 'use_ilu_precond': True})
        self.assertEqual(coupling_sellar.discipline_wrapp.discipline.max_mda_iter, 20)
        self.assertEqual(coupling_sellar.discipline_wrapp.discipline.use_lu_fact, True)

        self.assertEqual(
            coupling_sellar.discipline_wrapp.discipline.inner_mdas[0].mda_sequence[0].epsilon0, 1e-04)
        self.assertEqual(
            coupling_sellar.discipline_wrapp.discipline.inner_mdas[0].mda_sequence[1].mda_sequence[
                0].epsilon0, 1e-04)
        self.assertEqual(
            coupling_sellar.discipline_wrapp.discipline.inner_mdas[0].mda_sequence[1].mda_sequence[
                1].epsilon0, 1e-04)

        self.assertEqual(
            coupling_sellar.discipline_wrapp.discipline.inner_mdas[0].mda_sequence[1].mda_sequence[
                1].over_relaxation_factor, 0.85)

        self.exec_eng.execute()

        # run check_jac to test MDO numerical parameters
        dump_jac_path = join(dirname(__file__), 'jac_sellar_test_51.pkl')
        self.file_to_del.append(dump_jac_path)
        coupling_sellar.discipline_wrapp.discipline.check_jacobian(
            threshold=1.0e-7, dump_jac_path=dump_jac_path)

        self.assertEqual(coupling_sellar.discipline_wrapp.discipline.linear_solver, 'LGMRES')
        self.assertEqual(coupling_sellar.discipline_wrapp.discipline.linear_solver_tolerance, 1e-10)
        self.assertEqual(coupling_sellar.discipline_wrapp.discipline.linear_solver_settings, {
            'max_iter': 600, 'use_ilu_precond': True})


# =========================================================================


if '__main__' == __name__:
    cls = TestStructuringInputs()
    cls.setUp()
    cls.test_01_configure_discipline_with_setup_sos_discipline()
