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

"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
"""

import unittest
from numpy import array, arange, ones
import pandas as pd
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from os.path import join, dirname
from os import remove
from pathlib import Path
from time import sleep


class TestAnalyticGradients(unittest.TestCase):
    """
    Class to test analytic gradients of Sellar optim case
    """

    def setUp(self):
        self.study_name = 'optim'
        self.ns = f'{self.study_name}'
        self.c_name = "SellarCoupling"
        self.sc_name = "SellarOptimScenario"
        self.repo = 'sos_trades_core.sos_processes.test'
        self.proc_name = 'test_sellar_coupling'
        dspace_dict = {'variable': ['x', 'z', 'y_1', 'y_2'],
                       'value': [[1.], [5., 2.], [1.], [1.]],
                       'lower_bnd': [[0.], [-10., 0.], [-100.], [-100.]],
                       'upper_bnd': [[10.], [10., 10.], [100.], [100.]],
                       'enable_variable': [True, True, True, True],
                       'activated_elem': [[True], [True, True], [True], [True]]}
        self.dspace = pd.DataFrame(dspace_dict)
        self.file_to_del = []

    def tearDown(self):
        for file in self.file_to_del:
            if Path(file).is_file():
                remove(file)
                sleep(0.5)

    def test_1_check_analytic_gradients_simple_sellar(self):

        print("\n Test 1 : Check analytic gradients on standard Sellar")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id=self.proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.c_name}.chain_linearize'] = True
        values_dict[f'{self.ns}.{self.c_name}.x'] = array([2.])
        values_dict[f'{self.ns}.{self.c_name}.y_1'] = array([2.])
        values_dict[f'{self.ns}.{self.c_name}.y_2'] = array([2.])
        values_dict[f'{self.ns}.{self.c_name}.z'] = array([2., 2.])
        values_dict[f'{self.ns}.{self.c_name}.Sellar_Problem.local_dv'] = array([local_dv])

        exec_eng.load_study_from_input_dict(values_dict)

        for disc in exec_eng.root_process.sos_disciplines[0].sos_disciplines:
            with self.assertLogs('gemseo.utils.derivatives.derivatives_approx', level='INFO') as cm:
                disc.check_jacobian(threshold=1.0e-7)
            self.assertEqual(
                cm.output[-1], f'INFO:gemseo.utils.derivatives.derivatives_approx:Linearization of MDODiscipline: {disc.sos_name} is correct.', msg=cm.output)
        exec_eng.root_process.sos_disciplines[0].check_jacobian(
            threshold=1.0e-7, linearization_mode='adjoint')

    def test_2_check_analytic_gradients_sellar_new_types(self):

        print("\n Test 2 : Check analytic gradients on Sellar new types ")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id='test_sellar_coupling_new_types')

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        df = pd.DataFrame({'years': arange(1, 5)})
        df['value'] = 2.0

        dict_x = {'years': arange(1, 5), 'value': 2 * ones(4)}

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.c_name}.chain_linearize'] = True
        values_dict[f'{self.ns}.{self.c_name}.x'] = dict_x
        values_dict[f'{self.ns}.{self.c_name}.y_1'] = df.copy()
        values_dict[f'{self.ns}.{self.c_name}.y_2'] = df.copy()
        values_dict[f'{self.ns}.{self.c_name}.z'] = array([2., 2.])
        values_dict[f'{self.ns}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv

        exec_eng.load_study_from_input_dict(values_dict)

        for disc in exec_eng.root_process.sos_disciplines[0].sos_disciplines:
            with self.assertLogs('gemseo.utils.derivatives.derivatives_approx', level='INFO') as cm:
                disc.check_jacobian(threshold=1.0e-7)
            self.assertEqual(
                cm.output[-1], f'INFO:gemseo.utils.derivatives.derivatives_approx:Linearization of MDODiscipline: {disc.sos_name} is correct.', msg='\n'.join(cm.output))
        exec_eng.root_process.sos_disciplines[0].check_jacobian(
            threshold=1.0e-7, linearization_mode='adjoint')

    def test_3_optim_scenario_execution_mdf_with_user_mode(self):
        print("\n Test 3 : Sellar optim solution check with MDF formulation between user and finite diff \n \
                    compare user option with theory and compare with finite differences")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id='test_sellar_opt')

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        disc_dict = {}
        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 100
        disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "NLOPT_MMA"
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = self.dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'MDF'
        disc_dict[f'{self.ns}.SellarOptimScenario.differentiation_method'] = 'user'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = [
            f'c_1', f'c_2']

        disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-5,
                                                                    "ineq_tolerance": 1e-5,
                                                                    "normalize_design_space": False}
        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.y_1'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.y_2'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.z'] = array([1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.Sellar_Problem.local_dv'] = local_dv

        exec_eng.load_study_from_input_dict(values_dict)
        exec_eng.execute()

        # retrieve discipline to check the result...
        opt_disc = exec_eng.dm.get_disciplines_with_name(
            "optim." + self.sc_name)[0]

        # check optimal x and f
        sellar_obj_opt = 3.18339395 + local_dv
        self.assertAlmostEqual(
            sellar_obj_opt, opt_disc.optimization_result.f_opt[0], 4, msg="Wrong objective value")
        exp_x = array([8.45997174e-15, 1.97763888, 0.0])
        for x, x_th in zip(opt_disc.optimization_result.x_opt, exp_x):
            self.assertAlmostEqual(x, x_th, delta=1.0e-4,
                                   msg="Wrong optimal x solution")

        exec_eng_fd = ExecutionEngine(self.study_name)
        factory = exec_eng_fd.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id='test_sellar_opt')

        exec_eng_fd.factory.set_builders_to_coupling_builder(builder)

        exec_eng_fd.configure()

        # -- set up disciplines in Scenario
        disc_dict = {}
        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 100
        disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "NLOPT_MMA"
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = self.dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'MDF'
        disc_dict[f'{self.ns}.SellarOptimScenario.differentiation_method'] = 'finite_differences'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = [
            f'c_1', f'c_2']

        disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-5,
                                                                    "ineq_tolerance": 1e-5,
                                                                    "normalize_design_space": False}
        exec_eng_fd.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.y_1'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.y_2'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.z'] = array([1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.Sellar_Problem.local_dv'] = local_dv

        exec_eng_fd.load_study_from_input_dict(values_dict)
        exec_eng_fd.execute()

        # retrieve discipline to check the result...
        opt_disc_fd = exec_eng_fd.dm.get_disciplines_with_name(
            "optim." + self.sc_name)[0]

        # check optimal x and f
        self.assertAlmostEqual(
            opt_disc_fd.optimization_result.f_opt[0], opt_disc.optimization_result.f_opt[0], 4, msg="Wrong objective value")
        for x, x_fd in zip(opt_disc.optimization_result.x_opt, opt_disc_fd.optimization_result.x_opt):
            self.assertAlmostEqual(x, x_fd, delta=1.0e-4,
                                   msg="Wrong optimal x solution")

    def _test_4_optim_scenario_execution_idf_with_user_mode(self):
        # optim test with IDF formulation is not running anymore
        # since the data storage refactoring into local_data instead of
        # datamanager

        print("\n Test 4 : Sellar optim solution check with IDF formulation between user and finite diff \n \
                    compare user option with theory and compare with finite differences")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id='test_sellar_opt')

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        disc_dict = {}
        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 200
        disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "NLOPT_SLSQP"
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = self.dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'IDF'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = [
            f'c_1', f'c_2']
        disc_dict[f'{self.ns}.SellarOptimScenario.differentiation_method'] = 'user'
        disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-6,
                                                                    "ineq_tolerance": 1e-6,
                                                                    "normalize_design_space": True}
        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.y_1'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.y_2'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.z'] = array([1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.Sellar_Problem.local_dv'] = local_dv

        exec_eng.load_study_from_input_dict(values_dict)

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ optim',
                       f'\t|_ {self.sc_name}',
                       '\t\t|_ Sellar_Problem',
                       '\t\t|_ Sellar_2',
                       '\t\t|_ Sellar_1']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        exec_eng.execute()

        # retrieve discipline to check the result...
        opt_disc = exec_eng.dm.get_disciplines_with_name(
            "optim." + self.sc_name)[0]

        # check optimal x and f
        sellar_obj_opt = 3.18 + local_dv
        self.assertAlmostEqual(
            sellar_obj_opt, opt_disc.optimization_result.f_opt, 4, msg="Wrong objective value")
        exp_x = array([1.6653e-16, 2.1339, 0., 3.16, 3.911598])
        for x, x_th in zip(opt_disc.optimization_result.x_opt, exp_x):
            self.assertAlmostEqual(x, x_th, delta=1.0e-4,
                                   msg="Wrong optimal x solution")

        exec_eng_fd = ExecutionEngine(self.study_name)
        factory = exec_eng_fd.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id='test_sellar_opt')

        exec_eng_fd.factory.set_builders_to_coupling_builder(builder)

        exec_eng_fd.configure()

        # -- set up disciplines in Scenario
        disc_dict = {}
        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 200
        disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "NLOPT_SLSQP"
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = self.dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'IDF'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = [
            f'c_1', f'c_2']
        disc_dict[f'{self.ns}.SellarOptimScenario.differentiation_method'] = 'finite_differences'
        disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-6,
                                                                    "ineq_tolerance": 1e-6,
                                                                    "normalize_design_space": True}

        exec_eng_fd.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.y_1'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.y_2'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.z'] = array([1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.Sellar_Problem.local_dv'] = local_dv

        exec_eng_fd.load_study_from_input_dict(values_dict)
        exec_eng_fd.execute()

        # retrieve discipline to check the result...
        opt_disc_fd = exec_eng_fd.dm.get_disciplines_with_name(
            "optim." + self.sc_name)[0]

        # check optimal x and f
        self.assertAlmostEqual(
            opt_disc_fd.optimization_result.f_opt, opt_disc.optimization_result.f_opt, 4, msg="Wrong objective value")
        for x, x_fd in zip(opt_disc.optimization_result.x_opt, opt_disc_fd.optimization_result.x_opt):
            self.assertAlmostEqual(x, x_fd, delta=1.0e-4,
                                   msg="Wrong optimal x solution")

    def test_5_optim_scenario_execution_disciplinaryopt_with_user_mode(self):
        print("\n Test 5 : Sellar optim solution check with DisciplinaryOpt formulation between user \
                 and compare with MDf with user method ")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name_discopt = 'test_sellar_opt_discopt'
        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id=proc_name_discopt)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up design space
        dspace_dict = {'variable': ['x', 'z'],
                       'value': [[1.], [5., 2.]],
                       'lower_bnd': [[0.], [-10., 0.]],
                       'upper_bnd': [[10.], [10., 10.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True, True]]}
        dspace = pd.DataFrame(dspace_dict)

        # -- set up disciplines in Scenario
        disc_dict = {}
        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 200
        disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "NLOPT_SLSQP"
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.differentiation_method'] = 'user'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = [
            'c_1', 'c_2']

        disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-7,
                                                                    "ineq_tolerance": 1e-7,
                                                                    "normalize_design_space": False}

        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.z'] = array([
                                                                         1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)
        exec_eng.execute()

        opt_disc = exec_eng.dm.get_disciplines_with_name(
            "optim." + self.sc_name)[0]

        exec_eng_mdf = ExecutionEngine(self.study_name)
        factory = exec_eng_mdf.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id='test_sellar_opt')

        exec_eng_mdf.factory.set_builders_to_coupling_builder(builder)

        exec_eng_mdf.configure()

        # -- set up disciplines in Scenario
        disc_dict = {}
        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 100
        disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "NLOPT_MMA"
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = self.dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'MDF'
        disc_dict[f'{self.ns}.SellarOptimScenario.differentiation_method'] = 'user'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = [
            f'c_1', f'c_2']

        disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-5,
                                                                    "ineq_tolerance": 1e-5,
                                                                    "normalize_design_space": False}
        exec_eng_mdf.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.y_1'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.y_2'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.z'] = array([1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.Sellar_Problem.local_dv'] = local_dv

        exec_eng_mdf.load_study_from_input_dict(values_dict)
        exec_eng_mdf.execute()

        # retrieve discipline to check the result...
        opt_disc_mdf = exec_eng_mdf.dm.get_disciplines_with_name(
            "optim." + self.sc_name)[0]

        # check optimal x and f
        self.assertAlmostEqual(
            opt_disc_mdf.optimization_result.f_opt, opt_disc.optimization_result.f_opt, delta=1.0e-4, msg="Wrong objective value")
        for x, x_fd in zip(opt_disc.optimization_result.x_opt, opt_disc_mdf.optimization_result.x_opt):
            self.assertAlmostEqual(x, x_fd, delta=1.0e-4,
                                   msg="Wrong optimal x solution")

    def _test_6_optim_scenario_execution_mdf_with_different_linearization_mode(self):
        # optim test with MDF formulation is not running anymore
        # since the data storage refactoring into local_data instead of datamanager
        # to fix it, we need to add the datamanager update with local data at the end of _run method of MDAChain
        #
        # SECTION TO ADD IN MDAChain._run (gemseo/mda/mda_chain.py):
        #         local_data_sos = self.disciplines[0]._convert_array_into_new_type(
        #             self.local_data)
        #         self.disciplines[0].dm.set_values_from_dict(local_data_sos)

        print("\n Test 3 : Sellar optim solution check with MDF formulation between user and finite diff \n \
                    compare user option with theory and compare with finite differences")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id='test_sellar_opt')

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        disc_dict = {}
        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 100
        disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "NLOPT_MMA"
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = self.dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'MDF'
        disc_dict[f'{self.ns}.SellarOptimScenario.differentiation_method'] = 'user'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = [
            f'c_1', f'c_2']

        disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-5,
                                                                    "ineq_tolerance": 1e-5,
                                                                    "normalize_design_space": False}
        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.y_1'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.y_2'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.z'] = array([1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.configure()

        disc_sellar_problem = exec_eng.dm.get_disciplines_with_name(
            f'{self.ns}.{self.sc_name}.Sellar_Problem')[0]
        disc_sellar_problem.linearization_mode = 'finite_differences'

        exec_eng.execute()

        for disc in exec_eng.root_process.sos_disciplines:

            if disc.sos_name == 'Sellar_problem':
                self.assertEqual(disc.linearization_mode, 'finite_differences')
            else:
                self.assertEqual(disc.linearization_mode, 'auto')
        # retrieve discipline to check the result...
        opt_disc = exec_eng.dm.get_disciplines_with_name(
            "optim." + self.sc_name)[0]

        # check optimal x and f
        sellar_obj_opt = 3.18339395 + local_dv
        self.assertAlmostEqual(
            sellar_obj_opt, opt_disc.optimization_result.f_opt, 4, msg="Wrong objective value")
        exp_x = array([8.45997174e-15, 1.97763888, 0.0])
        for x, x_th in zip(opt_disc.optimization_result.x_opt, exp_x):
            self.assertAlmostEqual(x, x_th, delta=1.0e-4,
                                   msg="Wrong optimal x solution")

    def test_7_optim_scenario_execution_discopt_with_different_linearization_mode(self):
        print("\n Test 3 : Sellar optim solution check with MDF formulation between user and finite diff \n \
                    compare user option with theory and compare with finite differences")

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name_discopt = 'test_sellar_opt_discopt'
        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id=proc_name_discopt)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up design space
        dspace_dict = {'variable': ['x', 'z'],
                       'value': [[1.], [5., 2.]],
                       'lower_bnd': [[0.], [-10., 0.]],
                       'upper_bnd': [[10.], [10., 10.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True, True]]}
        dspace = pd.DataFrame(dspace_dict)

        # -- set up disciplines in Scenario
        disc_dict = {}
        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 200
        disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "NLOPT_SLSQP"
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.differentiation_method'] = 'user'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = [
            'c_1', 'c_2']

        disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-7,
                                                                    "ineq_tolerance": 1e-7,
                                                                    "normalize_design_space": False}

        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.z'] = array([
                                                                         1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.configure()

        disc_sellar_problem = exec_eng.dm.get_disciplines_with_name(
            f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem')[0]
        disc_sellar_problem.linearization_mode = 'finite_differences'

        exec_eng.execute()

        for disc in exec_eng.root_process.sos_disciplines[0].sos_disciplines[0].sos_disciplines:

            if disc.sos_name == 'Sellar_Problem':
                self.assertEqual(disc.linearization_mode, 'finite_differences')
            else:
                self.assertEqual(disc.linearization_mode, 'auto')
        opt_disc = exec_eng.dm.get_disciplines_with_name(
            "optim." + self.sc_name)[0]

        sellar_obj_opt = 3.18339395 + local_dv
        self.assertAlmostEqual(
            sellar_obj_opt, opt_disc.optimization_result.f_opt, 4, msg="Wrong objective value")
        exp_x = array([8.45997174e-15, 1.97763888, 0.0])
        for x, x_th in zip(opt_disc.optimization_result.x_opt, exp_x):
            self.assertAlmostEqual(x, x_th, delta=1.0e-4,
                                   msg="Wrong optimal x solution")

    def test_8_check_analytic_gradients_sellar_wo_new_types_with_mdaquasinewton(self):

        print("\n Test 8 : Check analytic gradients on with MDAQuasiNewton (without Sellar new types) ")
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
        values_dict[f'{self.ns}.{self.c_name}.sub_mda_class'] = "MDAQuasiNewton"
        values_dict[f'{self.ns}.{self.c_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.c_name}.y_1'] = array([1.])
        values_dict[f'{self.ns}.{self.c_name}.y_2'] = array([1.])
        values_dict[f'{self.ns}.{self.c_name}.z'] = array([1., 1.])
        values_dict[f'{self.ns}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)
        exec_eng.execute()

        obj, y_1, y_2 = exec_eng.root_process.sos_disciplines[0].get_sosdisc_outputs([
                                                                                     "obj", "y_1", "y_2"])

        obj_ref = array([14.32662157])
        y_1_ref = array([2.29689011])
        y_2_ref = array([3.51554944])

        self.assertAlmostEqual(obj_ref, obj.real, delta=1e-5)
        self.assertAlmostEqual(y_1_ref, y_1.real, delta=1e-5)
        self.assertAlmostEqual(y_2_ref, y_2.real, delta=1e-5)

    def test_9_check_analytic_gradients_sellar_wo_new_types_with_mdanewtonraphson(self):

        print("\n Test 9 : Check analytic gradients on with MDANewtonRaphson (without Sellar new types) ")
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
        values_dict[f'{self.ns}.{self.c_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.c_name}.y_1'] = array([1.])
        values_dict[f'{self.ns}.{self.c_name}.y_2'] = array([1.])
        values_dict[f'{self.ns}.{self.c_name}.z'] = array([1., 1.])
        values_dict[f'{self.ns}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        obj, y_1, y_2 = exec_eng.root_process.sos_disciplines[0].get_sosdisc_outputs([
                                                                                     "obj", "y_1", "y_2"])

        obj_ref = array([14.32662157])
        y_1_ref = array([2.29689011])
        y_2_ref = array([3.51554944])

        self.assertAlmostEqual(obj_ref, obj.real, delta=1e-5)
        self.assertAlmostEqual(y_1_ref, y_1.real, delta=1e-5)
        self.assertAlmostEqual(y_2_ref, y_2.real, delta=1e-5)

    def test_10_check_analytic_gradients_sellar_new_types_with_mdanewtonraphson(self):

        print("\n Test 10 : Check analytic gradients on with MDANewtonRaphson (with Sellar new types) ")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id='test_sellar_coupling_new_types')

        exec_eng.factory.set_builders_to_coupling_builder(builder)
        exec_eng.configure()

        df = pd.DataFrame({'years': arange(1, 5)})
        df['value'] = 2.0

        dict_x = {'years': arange(1, 5), 'value': 2 * ones(4)}

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.c_name}.chain_linearize'] = True
        values_dict[f'{self.ns}.{self.c_name}.tolerance'] = 1e-10
        values_dict[f'{self.ns}.{self.c_name}.sub_mda_class'] = "MDANewtonRaphson"
        values_dict[f'{self.ns}.{self.c_name}.x'] = dict_x
        values_dict[f'{self.ns}.{self.c_name}.y_1'] = df.copy()
        values_dict[f'{self.ns}.{self.c_name}.y_2'] = df.copy()
        values_dict[f'{self.ns}.{self.c_name}.z'] = array([2., 2.])
        values_dict[f'{self.ns}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        obj, y_1, y_2 = exec_eng.root_process.sos_disciplines[0].get_sosdisc_outputs([
                                                                                     "obj", "y_1", "y_2"])

        obj_ref = array([22.68435186 + 0.j])

        y_ref1 = df.copy()
        y_ref1['value'] = ones(len(arange(1, 5))) * 6.682971136716098 + 0.j

        y_ref2 = df.copy()
        y_ref2['value'] = ones(len(arange(1, 5))) * 6.585144316419511 + 0.j

        self.assertAlmostEqual(obj_ref, obj, delta=1e-5)
        self.assertListEqual(
            list(y_ref1['value'].values), list(y_1['value'].values))
        self.assertListEqual(
            list(y_ref2['value'].values), list(y_2['value'].values))

    def test_11_optim_scenario_execution_discopt_with_mode_adjoint(self):
        print("\n Test 11 : Sellar optim solution check with MDF formulation between user and finite diff \n \
                    compare user option with theory and compare with finite differences")

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name_discopt = 'test_sellar_opt_discopt'
        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id=proc_name_discopt)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up design space
        dspace_dict = {'variable': ['x', 'z'],
                       'value': [[1.], [5., 2.]],
                       'lower_bnd': [[0.], [-10., 0.]],
                       'upper_bnd': [[10.], [10., 10.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True, True]]}
        dspace = pd.DataFrame(dspace_dict)

        # -- set up disciplines in Scenario
        disc_dict = {}
        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 200
        disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "NLOPT_SLSQP"
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.differentiation_method'] = 'user'
        disc_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.linearization_mode'] = 'adjoint'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = [
            'c_1', 'c_2']

        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.z'] = array([
                                                                         1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        values_dict[f'{self.ns}.{self.sc_name}.algo_options'] = {'xtol_rel': 1e-08, 'normalize_design_space': True, 'xtol_abs': 1e-14, 'ftol_rel': 1e-08, 'ftol_abs': 1e-14,
         'max_iter': 999, 'max_time': 0.0, 'ctol_abs': 1e-06, 'eq_tolerance': 0.01, 'ineq_tolerance': 0.0001,
         'init_step': 0.25}

        exec_eng.load_study_from_input_dict(values_dict)
        exec_eng.execute()

        opt_disc = exec_eng.dm.get_disciplines_with_name(
            "optim." + self.sc_name)[0]

        sellar_obj_opt = 3.18339395 + local_dv
        self.assertAlmostEqual(
            sellar_obj_opt, opt_disc.optimization_result.f_opt[0], 4, msg="Wrong objective value")
        exp_x = array([8.45997174e-15, 1.97763888, 0.0])
        for x, x_th in zip(opt_disc.optimization_result.x_opt, exp_x):
            self.assertAlmostEqual(x, x_th, delta=1.0e-4,
                                   msg="Wrong optimal x solution")

        mda_disc = exec_eng.dm.get_disciplines_with_name(
            f'{self.ns}.{self.sc_name}.{self.c_name}')[0]
        n_calls_linearize_adjoint = mda_disc.n_calls_linearize
        n_calls_adjoint = mda_disc.n_calls
        self.assertLessEqual(n_calls_adjoint, 48)
        self.assertLessEqual(n_calls_linearize_adjoint, 24)

    def test_12_test_option_load_dump_jac(self):

        print("\n Test 12 : Test options load and dump jac on standard Sellar")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id=self.proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        print('\n in test optim scenario')

        # -- set up disciplines in Scenario

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.c_name}.chain_linearize'] = True
        values_dict[f'{self.ns}.{self.c_name}.x'] = array([2.])
        values_dict[f'{self.ns}.{self.c_name}.y_1'] = array([2.])
        values_dict[f'{self.ns}.{self.c_name}.y_2'] = array([2.])
        values_dict[f'{self.ns}.{self.c_name}.z'] = array([2., 2.])
        values_dict[f'{self.ns}.{self.c_name}.Sellar_Problem.local_dv'] = array([local_dv])
        exec_eng.load_study_from_input_dict(values_dict)

        dump_jac_path = join(dirname(__file__), 'jac_sellar.pkl')
        for disc in exec_eng.root_process.sos_disciplines[0].sos_disciplines:
            # First dump the jacobian
            disc.check_jacobian(threshold=1.0e-7, dump_jac_path=dump_jac_path)
            # Then check the load
            disc.check_jacobian(threshold=1.0e-7, load_jac_path=dump_jac_path)

        self.file_to_del.append(dump_jac_path)

        exec_eng.root_process.sos_disciplines[0].check_jacobian(
            threshold=1.0e-7, linearization_mode='adjoint', dump_jac_path=dump_jac_path)
        exec_eng.root_process.sos_disciplines[0].check_jacobian(
            threshold=1.0e-7, linearization_mode='adjoint', load_jac_path=dump_jac_path)

    def test_13_check_analytic_gradients_adjoint_sellar_new_types_nowarmstart(self):

        print(
            "\n Test 13 : Check adjoint gradients on Sellar new types with warm_start=False")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id='test_sellar_coupling_new_types')

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        df = pd.DataFrame({'years': arange(1, 5)})
        df['value'] = 2.0

        dict_x = {'years': arange(1, 5), 'value': 2 * ones(4)}

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.c_name}.chain_linearize'] = False
        values_dict[f'{self.ns}.{self.c_name}.warm_start'] = False
        values_dict[f'{self.ns}.{self.c_name}.sub_mda_class'] = 'MDAGaussSeidel'
        values_dict[f'{self.ns}.{self.c_name}.x'] = dict_x
        values_dict[f'{self.ns}.{self.c_name}.y_1'] = df.copy()
        values_dict[f'{self.ns}.{self.c_name}.y_2'] = df.copy()
        values_dict[f'{self.ns}.{self.c_name}.z'] = array([2., 2.])
        values_dict[f'{self.ns}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.root_process.sos_disciplines[0].check_jacobian(
            threshold=1.0e-7, linearization_mode='adjoint',
            derr_approx='complex_step', step=1e-15)

    def test_14_check_analytic_gradients_adjoint_sellar_new_types_with_warmstart(self):

        print(
            "\n Test 14 : Check adjoint gradients on Sellar new types with warm_start=True")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id='test_sellar_coupling_new_types')

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        df = pd.DataFrame({'years': arange(1, 5)})
        df['value'] = 2.0

        dict_x = {'years': arange(1, 5), 'value': 2 * ones(4)}

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.c_name}.chain_linearize'] = False
        values_dict[f'{self.ns}.{self.c_name}.warm_start'] = True
        values_dict[f'{self.ns}.{self.c_name}.tolerance'] = 1.0e-8
        # starts from the 4th iteration
        values_dict[f'{self.ns}.{self.c_name}.warm_start_threshold'] = 1.e3
        values_dict[f'{self.ns}.{self.c_name}.sub_mda_class'] = 'MDAGaussSeidel'
        values_dict[f'{self.ns}.{self.c_name}.x'] = dict_x
        values_dict[f'{self.ns}.{self.c_name}.y_1'] = df.copy()
        values_dict[f'{self.ns}.{self.c_name}.y_2'] = df.copy()
        values_dict[f'{self.ns}.{self.c_name}.z'] = array([2., 2.])
        values_dict[f'{self.ns}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)
        exec_eng.display_treeview_nodes(display_variables=True)

        exec_eng.root_process.sos_disciplines[0].check_jacobian(
            threshold=1.0e-7, linearization_mode='adjoint',
            derr_approx='complex_step', step=1e-15)

    def test_15_check_analytic_gradients_adjoint_sellar_new_types_with_warmstart_newtonraphson(self):

        print("\n Test 15 : Check adjoint gradients on Sellar new types with warm_start=True newton raphson")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id='test_sellar_coupling_new_types')

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        df = pd.DataFrame({'years': arange(1, 5)})
        df['value'] = 2.0

        dict_x = {'years': arange(1, 5), 'value': 2 * ones(4)}

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.c_name}.chain_linearize'] = False
        values_dict[f'{self.ns}.{self.c_name}.warm_start'] = False
        values_dict[f'{self.ns}.{self.c_name}.tolerance'] = 1.0e-8
        # starts from the 4th iteration
        values_dict[f'{self.ns}.{self.c_name}.warm_start_threshold'] = 1.e3
        values_dict[f'{self.ns}.{self.c_name}.sub_mda_class'] = 'MDANewtonRaphson'
        values_dict[f'{self.ns}.{self.c_name}.x'] = dict_x
        values_dict[f'{self.ns}.{self.c_name}.y_1'] = df.copy()
        values_dict[f'{self.ns}.{self.c_name}.y_2'] = df.copy()
        values_dict[f'{self.ns}.{self.c_name}.z'] = array([2., 2.])
        values_dict[f'{self.ns}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.root_process.sos_disciplines[0].check_jacobian(
            threshold=1.0e-7, linearization_mode='adjoint',
            derr_approx='finite_differences', step=1e-15)


if '__main__' == __name__:
    cls = TestAnalyticGradients()
    cls.setUp()
    cls.test_7_optim_scenario_execution_discopt_with_different_linearization_mode()
