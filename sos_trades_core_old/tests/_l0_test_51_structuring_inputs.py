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
from time import sleep, time
from pathlib import Path
import platform

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.execution_engine.sos_coupling import SoSCoupling
from sos_trades_core.sos_processes.test.test_configure_process.usecase import Study as study_core
from sos_trades_core.sos_processes.test.test_configure_process.usecase_import_study import Study as study_core_import_study
from sos_trades_core.sos_processes.test.test_sellar_opt_discopt.usecase import Study as study_sellar_opt

from tempfile import gettempdir
from copy import copy, deepcopy
from os.path import join, dirname
from os import remove
import os


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
        self.assertListEqual(list(self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Disc1')[0]._structuring_variables.keys()), ['AC_list', 'cache_type', 'cache_file_path', 'debug_mode', 'dyn_input_2'])

        self.exec_eng.load_study_from_input_dict(full_values_dict)
        print(self.exec_eng.display_treeview_nodes())

        full_values_dict[self.study_name + '.x'] = 2

        self.exec_eng.dm.set_values_from_dict(full_values_dict)
        disc_to_conf = self.exec_eng.root_process.get_disciplines_to_configure()
        self.assertListEqual(disc_to_conf, [])
        self.assertListEqual(list(self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Disc1')[0]._structuring_variables.keys()), ['AC_list', 'cache_type', 'cache_file_path', 'debug_mode', 'dyn_input_2'])

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

    def test_02_configure_core_process(self):

        builder_process = self.exec_eng.factory.get_builder_from_process(
            self.repo, 'test_configure_process')
        self.exec_eng.factory.set_builders_to_coupling_builder(
            builder_process)

        self.exec_eng.configure()

        self.exec_eng.display_treeview_nodes()
        usecase = study_core(execution_engine=self.exec_eng)
        usecase.study_name = self.namespace
        values_dict = usecase.setup_usecase()

        full_values_dict = {}
        for values in values_dict:
            full_values_dict.update(values)

        t0 = time()
        self.exec_eng.load_study_from_input_dict(full_values_dict)
        self.exec_eng.execute()
        t1 = time() - t0
        # print(f'First configure time: {t1}\n')
        # FIRST CONFIGURE
        # Before: 0.04054093360900879
        # After configure refactoring: 0.033379316329956055
        # With structuring variables: 0.03693001174926758
        # With new compare_dict method: 0.03120565414428711

        for disc in self.exec_eng.dm.disciplines_dict.keys():
            self.assertTrue(self.exec_eng.dm.get_discipline(
                disc).get_configure_status())
        for disc in self.exec_eng.dm.disciplines_dict.keys():
            self.assertTrue(self.exec_eng.dm.get_discipline(
                disc).is_configured())

        full_values_dict[self.study_name + 
                         '.multi_scenarios.name_list'] = ['name_1', 'name_3', 'name_4']
        full_values_dict[self.study_name + 
                         '.multi_scenarios.z_dict'] = {'scenario_1': 1, 'scenario_2': 2, 'scenario_3': 4, 'scenario_4': 0}

        scenario_list = ['scenario_1', 'scenario_2',
                         'scenario_3', 'scenario_4']
        for scenario in scenario_list:
            full_values_dict[self.study_name + '.name_3.a'] = 1
            full_values_dict[self.study_name + '.name_4.a'] = 1
            full_values_dict[self.study_name + '.name_3.x'] = 2
            full_values_dict[self.study_name + '.name_4.x'] = 2
            full_values_dict[self.study_name + '.multi_scenarios.' + 
                             scenario + '.Disc1.name_3.b'] = 5
            full_values_dict[self.study_name + '.multi_scenarios.' + 
                             scenario + '.Disc1.name_4.b'] = 5

        t0 = time()
        self.exec_eng.load_study_from_input_dict(full_values_dict)
        t1 = time() - t0
        # print(f'Configure time after dm change: {t1}\n')
        # CONFIGURE WITH CHANGE IN DM
        # Before: 0.03207278251647949
        # After configure refactoring: 0.02335071563720703
        # With structuring variables : 0.02295012092590332
        # With new compare_dict method: 0.01555490493774414

        t0 = time()
        self.exec_eng.load_study_from_input_dict(full_values_dict)
        t1 = time() - t0
        # print(f'Configure time with no change: {t1}\n')
        # CONFIGURE WITH NO CHANGE IN DM
        # Before: 0.013063430786132812
        # After configure refactoring: 0.011217832565307617
        # With structuring variables : 0.004949092864990234
        # With new compare_dict method: 0.0

        self.exec_eng.execute()

    def _test_03_study_instanciation_core_process(self):

        t0 = time()
        uc_cls = study_core_import_study()
        uc_cls.setup_usecase()
        t1 = time() - t0
        print(f'Setup usecase time: {t1}\n')
        # STUDY INSTANCIATION
        # Before: 0.6121478080749512
        # After: 0.5095164775848389
        # With new compare_dict method: 0.49619007110595703

        t0 = time()
        uc_cls = study_core_import_study()
        uc_cls.load_data()
        t1 = time() - t0
        print(f'Study instanciation and load data time: {t1}\n')
        # STUDY INSTANCIATION
        # Before: 0.07483434677124023
        # After: 0.03124070167541504
        # With new compare_dict method: 0.0469052791595459

    def test_04_SoSCoupling_structuring_variables(self):

        repo_discopt = 'sostrades_core.sos_processes.test'
        proc_name_discopt = 'test_sellar_opt_discopt'
        builder = self.exec_eng.factory.get_builder_from_process(repo=repo_discopt,
                                                                 mod_id=proc_name_discopt)

        self.exec_eng.factory.set_builders_to_coupling_builder(builder)

        self.exec_eng.configure()

        uc_cls = study_sellar_opt()
        dict_values = uc_cls.setup_usecase()

        self.exec_eng.load_study_from_input_dict(dict_values[0])

        # check with defaults
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.SellarOptimScenario.SellarCoupling.tolerance'), SoSCoupling.DESC_IN['tolerance']['default'])

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.SellarOptimScenario.SellarCoupling.linear_solver_MDA_options'), SoSCoupling.DESC_IN['linear_solver_MDA_options']['default'])

        # check with custom values
        max_iter = 10
        mda_tolerance = 1e-08
        linear_solver_tol = 1.0e-5
        linear_solver_MDA_options = {
            'max_iter': max_iter, 'tol': linear_solver_tol}
        dict_values = {'MyCase.SellarOptimScenario.SellarCoupling.tolerance': mda_tolerance,
                       'MyCase.SellarOptimScenario.SellarCoupling.linear_solver_MDA_options': linear_solver_MDA_options,
                       'MyCase.SellarOptimScenario.SellarCoupling.sub_mda_class': "MDANewtonRaphson"}

        self.exec_eng.load_study_from_input_dict(dict_values)

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.SellarOptimScenario.SellarCoupling.tolerance'), 1e-08)

        coupling_disc = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.SellarOptimScenario.SellarCoupling')[0]

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.SellarOptimScenario.SellarCoupling.tolerance'), coupling_disc.tolerance)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.SellarOptimScenario.SellarCoupling.tolerance'), mda_tolerance)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.SellarOptimScenario.SellarCoupling.tolerance'), coupling_disc.sub_mda_list[0].tolerance)

        # check linear solver options for MDANewtonRaphson (gradient based, thus with linear solver options)
        # in sostrades, both "tol" and and "max_iter" are filled, in GEMSEO the
        # tolerance is filled in a class variable "linear_solver_tolerance"
        linear_solver_options_gemseo = copy(
            coupling_disc.sub_mda_list[0].linear_solver_options)

        if platform.system() == 'Windows':
            linear_solver_options_ref = {
                'max_iter': 10, 'use_ilu_precond': False}
        else:
            linear_solver_options_ref = {
                'max_iter': 10, 'preconditioner_type': 'gasm', 'solver_type': 'gmres'}

        self.assertDictEqual(linear_solver_options_ref,
                             linear_solver_options_gemseo)

        linear_solver_options_dm = deepcopy(self.exec_eng.dm.get_value(
            'MyCase.SellarOptimScenario.SellarCoupling.linear_solver_MDA_options'))
        tol_dm = linear_solver_options_dm.pop('tol')
        linear_solver_tolerance_gemseo = coupling_disc.sub_mda_list[0].linear_solver_tolerance
        assert linear_solver_tolerance_gemseo == tol_dm
        assert linear_solver_tolerance_gemseo == linear_solver_tol

        # check options for MDAGaussSeidel (no gradients, thus no linear solver
        # options)
        mda_tolerance = 1e-09
        dict_values = {'MyCase.SellarOptimScenario.SellarCoupling.tolerance': mda_tolerance,
                       'MyCase.SellarOptimScenario.SellarCoupling.sub_mda_class': "MDAGaussSeidel",
                       'MyCase.SellarOptimScenario.SellarCoupling.linear_solver_MDA_options': SoSCoupling.DEFAULT_LINEAR_SOLVER_OPTIONS
                       }
        self.exec_eng.load_study_from_input_dict(dict_values)
        coupling_disc = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.SellarOptimScenario.SellarCoupling')[0]

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.SellarOptimScenario.SellarCoupling.tolerance'), coupling_disc.tolerance)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.SellarOptimScenario.SellarCoupling.tolerance'), mda_tolerance)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.SellarOptimScenario.SellarCoupling.tolerance'), coupling_disc.sub_mda_list[0].tolerance)

        linear_solver_options_gemseo = copy(
            coupling_disc.sub_mda_list[0].linear_solver_options)

        if platform.system() == 'Windows':
            linear_solver_options_ref = {
                'max_iter': 1000, 'use_ilu_precond': False}
        else:
            linear_solver_options_ref = {
                'max_iter': 1000, 'preconditioner_type': 'gasm', 'solver_type': 'gmres'}

        self.assertDictEqual(linear_solver_options_ref,
                             linear_solver_options_gemseo)

        linear_solver_options_dm = copy(self.exec_eng.dm.get_value(
            'MyCase.SellarOptimScenario.SellarCoupling.linear_solver_MDA_options'))
        linear_solver_tolerance_gemseo = coupling_disc.sub_mda_list[0].linear_solver_tolerance
        assert linear_solver_tolerance_gemseo == 1.0e-8  # GEMSEO default value
        # provided by SoSTrades
        assert linear_solver_options_dm == SoSCoupling.DEFAULT_LINEAR_SOLVER_OPTIONS

    def test_05_SoSCoupling_numerical_inputs(self):

        repo_discopt = 'sostrades_core.sos_processes.test'
        proc_name_discopt = 'test_sellar_opt_discopt'
        builder = self.exec_eng.factory.get_builder_from_process(repo=repo_discopt,
                                                                 mod_id=proc_name_discopt)

        self.exec_eng.factory.set_builders_to_coupling_builder(builder)

        self.exec_eng.load_study_from_input_dict({})

        coupling_inputs = {'sub_mda_class': {'type': 'string', 'default': 'MDAJacobi', 'possible_values': ['MDAJacobi', 'MDAGaussSeidel', 'MDANewtonRaphson', 'PureNewtonRaphson',
                                                                                                           'MDAQuasiNewton', 'GSNewtonMDA', 'GSPureNewtonMDA', 'GSorNewtonMDA', 'MDASequential', 'GSPureNewtonorGSMDA']},
                           'max_mda_iter': {'type': 'int', 'default': 30, 'possible_values': None},
                           'n_processes': {'type': 'int', 'default': 1, 'possible_values': None},
                           'chain_linearize': {'type': 'bool', 'default': False, 'possible_values': [True, False]},
                           'tolerance': {'type': 'float', 'default': 1e-06, 'possible_values': None},
                           'use_lu_fact': {'type': 'bool', 'default': False, 'possible_values': [True, False]},
                           'warm_start': {'type': 'bool', 'default': False, 'possible_values': [True, False]},
                           'acceleration': {'type': 'string', 'default': 'm2d', 'possible_values': ['m2d', 'secant', 'none']},
                           'warm_start_threshold': {'type': 'float', 'default':-1, 'possible_values': None},
                           'n_subcouplings_parallel': {'type': 'int', 'default': 1, 'possible_values': None},
                           'tolerance_gs': {'type': 'float', 'default': 10.0, 'possible_values': None},
                           'relax_factor': {'type': 'float', 'default': 0.99, 'possible_values': None},
                           'epsilon0': {'type': 'float', 'default': 1e-06, 'possible_values': None},
                           'linear_solver_MDO': {'type': 'string'},
                           'linear_solver_MDO_preconditioner': {'type': 'string'},
                           'linear_solver_MDO_options': {'type': 'dict', 'default': {'max_iter': 1000, 'tol': 1e-08}, 'possible_values': None},
                           'linear_solver_MDA': {'type': 'string'},
                           'linear_solver_MDA_preconditioner': {'type': 'string'},
                           'linear_solver_MDA_options': {'type': 'dict', 'default': {'max_iter': 1000, 'tol': 1e-08}, 'possible_values': None},
                           'group_mda_disciplines': {'type': 'bool', 'default': False, 'possible_values': [True, False]},
                           'authorize_self_coupled_disciplines': {'type': 'bool', 'possible_values': [True, False],
                                                                  'default': False},
                           'linearization_mode': {'type': 'string', 'default': 'auto', 'possible_values': ['auto', 'direct', 'adjoint', 'reverse', 'finite_differences', 'complex_step']},
                           'cache_type': {'type': 'string', 'default': 'None', 'possible_values': ['None', 'SimpleCache']},
                           'cache_file_path': {'type': 'string', 'default': '', 'possible_values': None},
                           'debug_mode': {'type': 'string', 'default': '',
                                          'possible_values': ["", "nan", "input_change", "linearize_data_change", "min_max_grad", "min_max_couplings", "all"], }
                           }

        if platform.system() == 'Windows':
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
                                                                       'BICG_PETSC', 'BCGS_PETSC', 'LGMRES', 'GMRES', 'BICG', 'QMR', 'BICGSTAB', 'DEFAULT']
            coupling_inputs['linear_solver_MDA']['possible_values'] = ['GMRES_PETSC', 'LGMRES_PETSC',
                                                                       'BICG_PETSC', 'BCGS_PETSC', 'LGMRES', 'GMRES', 'BICG', 'QMR', 'BICGSTAB', 'DEFAULT']
            coupling_inputs['linear_solver_MDO_preconditioner']['default'] = 'gasm'
            coupling_inputs['linear_solver_MDA_preconditioner']['default'] = 'gasm'
            coupling_inputs['linear_solver_MDO_preconditioner']['possible_values'] = [
                'None', 'jacobi', 'ilu', 'gasm']
            coupling_inputs['linear_solver_MDA_preconditioner']['possible_values'] = [
                'None', 'jacobi', 'ilu', 'gasm']

        # check numerical inputs of root_process coupling
        self.assertListEqual(list(coupling_inputs.keys()), list(
            self.exec_eng.root_process._data_in.keys()))

        for input_name, input_dict in coupling_inputs.items():
            for key, value in input_dict.items():
                self.assertEqual(
                    self.exec_eng.root_process._data_in[input_name][key], value)

        self.assertEqual(self.exec_eng.dm.get_data(
            'MyCase.SellarOptimScenario.SellarCoupling.tolerance_gs', 'default'), 10.0)
        self.assertEqual(self.exec_eng.dm.get_data(
            'MyCase.SellarOptimScenario.SellarCoupling.tolerance_gs', 'value'), 10.0)

        uc_cls = study_sellar_opt()
        uc_cls.study_name = 'MyCase'
        dict_values = uc_cls.setup_usecase()
        dict_values[0]['MyCase.SellarOptimScenario.SellarCoupling.sub_mda_class'] = 'GSorNewtonMDA'
        dict_values[0]['MyCase.SellarOptimScenario.SellarCoupling.max_mda_iter'] = 20
        dict_values[0]['MyCase.SellarOptimScenario.SellarCoupling.tolerance'] = 1e-3
        dict_values[0]['MyCase.SellarOptimScenario.SellarCoupling.relax_factor'] = 0.85
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
            self.exec_eng.root_process._data_in.keys()))

        for input_name, input_dict in coupling_inputs.items():
            for key, value in input_dict.items():
                self.assertEqual(
                    self.exec_eng.root_process._data_in[input_name][key], value)

        coupling_sellar = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.SellarOptimScenario.SellarCoupling')[0]

        self.assertEqual(coupling_sellar.linear_solver, 'LGMRES')
        self.assertEqual(coupling_sellar.tolerance, 1e-3)
        self.assertEqual(coupling_sellar.sub_mda_list[0].tolerance, 1e-03)
        self.assertEqual(coupling_sellar.linear_solver_tolerance, 1e-10)
        self.assertEqual(coupling_sellar.linear_solver_options, {
                         'max_iter': 600, 'use_ilu_precond': True})
        self.assertEqual(coupling_sellar.max_mda_iter, 20)
        self.assertEqual(coupling_sellar.use_lu_fact, True)

        self.assertEqual(
            coupling_sellar.sub_mda_list[0].mda_sequence[0].epsilon0, 1e-04)
        self.assertEqual(
            coupling_sellar.sub_mda_list[0].mda_sequence[1].mda_sequence[0].epsilon0, 1e-04)
        self.assertEqual(
            coupling_sellar.sub_mda_list[0].mda_sequence[1].mda_sequence[1].epsilon0, 1e-04)

        self.assertEqual(
            coupling_sellar.sub_mda_list[0].mda_sequence[1].mda_sequence[1].relax_factor, 0.85)

        self.exec_eng.execute()

        # run check_jac to test MDO numerical parameters
        dump_jac_path = join(dirname(__file__), 'jac_sellar_test_51.pkl')
        self.file_to_del.append(dump_jac_path)
        coupling_sellar.check_jacobian(
            threshold=1.0e-7, dump_jac_path=dump_jac_path)

        self.assertEqual(coupling_sellar.linear_solver, 'LGMRES')
        self.assertEqual(coupling_sellar.linear_solver_tolerance, 1e-10)
        self.assertEqual(coupling_sellar.linear_solver_options, {
                         'max_iter': 600, 'use_ilu_precond': True})


if '__main__' == __name__:

    cls = TestStructuringInputs()
    cls.setUp()
    cls.test_05_SoSCoupling_numerical_inputs()
