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
import logging
from logging import Handler
from time import time

from pandas._testing import assert_frame_equal

from gemseo.algos.doe.doe_factory import DOEFactory

"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
unit test for doe scenario
"""

import unittest
from numpy import array
import pandas as pd
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
import os
from os.path import dirname, join


class TestSoSDOEScenario(unittest.TestCase):

    def setUp(self):
        self.study_name = 'doe'
        self.ns = f'{self.study_name}'
        self.sc_name = "SellarDoeScenario"
        self.c_name = "SellarCoupling"
        dspace_dict = {'variable': ['x', 'z', 'y_1', 'y_2'],
                       'value': [[1.], [5., 2.], [1.], [1.]],
                       'lower_bnd': [[0.], [-10., 0.], [-100.], [-100.]],
                       'upper_bnd': [[10.], [10., 10.], [100.], [100.]],
                       'enable_variable': [True, True, True, True],
                       'activated_elem': [[True], [True, True], [True], [True]]}

        dspace_dict_optim = {'variable': ['x', 'z', 'y_1', 'y_2'],
                             'value': [[1.], [5., 2.], [1.], [1.]],
                             'lower_bnd': [[0.], [-10., 0.], [-100.], [-100.]],
                             'upper_bnd': [[10.], [10., 10.], [100.], [100.]],
                             'enable_variable': [True, True, True, True],
                             'activated_elem': [[True], [True, True], [True], [True]]}

        self.dspace = pd.DataFrame(dspace_dict)

        self.dspace_optim = pd.DataFrame(dspace_dict_optim)

        self.repo = 'sostrades_core.sos_processes.test'
        self.proc_name = 'test_sellar_doe'

    def test_1_doe_scenario_check_treeview(self):
        print("\n Test 1 : check configure and treeview")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id=self.proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        disc_dict = {}
        # DoE inputs
        disc_dict[f'{self.ns}.SellarDoeScenario.n_samples'] = 100
        # 'lhs', 'fullfact', ...
        disc_dict[f'{self.ns}.SellarDoeScenario.algo'] = "lhs"
        disc_dict[f'{self.ns}.SellarDoeScenario.design_space'] = self.dspace_optim
        disc_dict[f'{self.ns}.SellarDoeScenario.formulation'] = 'MDF'
        disc_dict[f'{self.ns}.SellarDoeScenario.objective_name'] = 'obj'
        # disc_dict[f'{self.ns}.SellarDoeScenario.ineq_constraints'] = [f'c_1', f'c_2']

        disc_dict[f'{self.ns}.SellarDoeScenario.algo_options'] = {'levels': 'None'
                                                                  }
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

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ {self.sc_name}',
                       '\t\t|_ Sellar_Problem',
                       '\t\t|_ Sellar_2',
                       '\t\t|_ Sellar_1']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        # XDSMize test

    #         exec_eng.root_process.xdsmize()
    # to visualize in an internet browser :
    # - download XDSMjs at https://github.com/OneraHub/XDSMjs and unzip
    # - replace existing xdsm.json inside by yours
    # - in the same folder, type in terminal 'python -m http.server 8080'
    # - open in browser http://localhost:8080/xdsm.html

    def test_2_doe_scenario_execution_lhs(self):
        print("\n Test 3 : Sellar doe solution with LHS algorithm")

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        repo_discopt = 'sostrades_core.sos_processes.test'
        proc_name_discopt = 'test_sellar_doe_discopt'
        builder = factory.get_builder_from_process(repo=repo_discopt,
                                                   mod_id=proc_name_discopt)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        disc_dict = {}
        # DoE inputs
        n_samples = 10
        disc_dict[f'{self.ns}.SellarDoeScenario.n_samples'] = n_samples
        # 'lhs', 'fullfact', ...
        disc_dict[f'{self.ns}.SellarDoeScenario.algo'] = "lhs"
        disc_dict[f'{self.ns}.SellarDoeScenario.design_space'] = self.dspace_optim
        disc_dict[f'{self.ns}.SellarDoeScenario.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{self.ns}.SellarDoeScenario.objective_name'] = 'obj'
        # disc_dict[f'{self.ns}.SellarDoeScenario.ineq_constraints'] = [f'c_1', f'c_2']

        disc_dict[f'{self.ns}.SellarDoeScenario.algo_options'] = {'levels': 'None'
                                                                  }
        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        # array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.z'] = array([
            1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.configure()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ {self.sc_name}',
                       f'\t\t|_ {self.c_name}',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1',
                       '\t\t\t|_ Sellar_Problem']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        exec_eng.execute()

        # retrieve discipline to check the result...
        doe_disc = exec_eng.dm.get_disciplines_with_name(
            "doe." + self.sc_name)[0]

        doe_disc_output = doe_disc.get_sosdisc_outputs()
        XY_pd = doe_disc_output['doe_ds_io']
        # print(XY_pd)
        X_pd = XY_pd['design_parameters']
        self.assertEqual(len(X_pd), n_samples)

    def test_3_doe_scenario_execution_fd_parallel(self):
        if os.name == 'nt':
            print("\n Test 03 : skipped, multi-proc not handled on windows")
        else:
            print("\n Test 03 : Sellar doe with FD in parallel execution")
            exec_eng = ExecutionEngine(self.study_name)
            factory = exec_eng.factory

            repo_discopt = 'sostrades_core.sos_processes.test'
            proc_name_discopt = 'test_sellar_doe_discopt'
            builder = factory.get_builder_from_process(repo=repo_discopt,
                                                       mod_id=proc_name_discopt)

            exec_eng.factory.set_builders_to_coupling_builder(builder)

            exec_eng.configure()
            n_samples = 10
            # -- set up disciplines in Scenario
            disc_dict = {}
            # DoE inputs
            disc_dict[f'{self.ns}.SellarDoeScenario.n_samples'] = n_samples
            # 'lhs', 'fullfact', ...
            disc_dict[f'{self.ns}.SellarDoeScenario.algo'] = "lhs"
            disc_dict[f'{self.ns}.SellarDoeScenario.design_space'] = self.dspace_optim
            disc_dict[f'{self.ns}.SellarDoeScenario.formulation'] = 'DisciplinaryOpt'
            disc_dict[f'{self.ns}.SellarDoeScenario.objective_name'] = 'obj'
            # disc_dict[f'{self.ns}.SellarDoeScenario.ineq_constraints'] = [f'c_1', f'c_2']

            disc_dict[f'{self.ns}.SellarDoeScenario.algo_options'] = {'levels': 'None'
                                                                      }

            # parallel inputs
            disc_dict[f'{self.ns}.SellarDoeScenario.parallel_options'] = {"parallel": True,
                                                                          "n_processes": 2,
                                                                          "use_threading": False,
                                                                          "wait_time_between_fork": 0}

            exec_eng.load_study_from_input_dict(disc_dict)

            # Sellar inputs
            local_dv = 10.
            values_dict = {}
            # array([1.])
            values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = array([1.])
            values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = array([1.])
            values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = array([1.])
            values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.z'] = array([
                1., 1.])
            values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
            exec_eng.load_study_from_input_dict(values_dict)

            exec_eng.configure()

            exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                           '|_ doe',
                           f'\t|_ {self.sc_name}',
                           f'\t\t|_ {self.c_name}',
                           '\t\t\t|_ Sellar_2',
                           '\t\t\t|_ Sellar_1',
                           '\t\t\t|_ Sellar_Problem']
            exp_tv_str = '\n'.join(exp_tv_list)
            exec_eng.display_treeview_nodes(True)
            assert exp_tv_str == exec_eng.display_treeview_nodes()

            exec_eng.execute()

            # retrieve discipline to check the result...
            # retrieve discipline to check the result...
            doe_disc = exec_eng.dm.get_disciplines_with_name(
                "doe." + self.sc_name)[0]

            doe_disc_output = doe_disc.get_sosdisc_outputs()
            XY_pd = doe_disc_output['doe_ds_io']
            # print(XY_pd)
            X_pd = XY_pd['design_parameters']
            self.assertEqual(len(X_pd), n_samples)

    def _test_4_test_options_full_fact(self):
        print("\n Test 04: Sellar doe solution check with DisciplinaryOpt formulation/ fullfact algo")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        repo_discopt = 'sostrades_core.sos_processes.test'
        proc_name_discopt = 'test_sellar_doe_discopt'
        builder = factory.get_builder_from_process(repo=repo_discopt,
                                                   mod_id=proc_name_discopt)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()
        n_samples = 10
        # -- set up disciplines in Scenario
        disc_dict = {}
        # DoE inputs
        disc_dict[f'{self.ns}.SellarDoeScenario.n_samples'] = n_samples
        # 'lhs', 'fullfact', ...
        disc_dict[f'{self.ns}.SellarDoeScenario.algo'] = "fullfact"
        disc_dict[f'{self.ns}.SellarDoeScenario.design_space'] = self.dspace_optim
        disc_dict[f'{self.ns}.SellarDoeScenario.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{self.ns}.SellarDoeScenario.objective_name'] = 'obj'
        # disc_dict[f'{self.ns}.SellarDoeScenario.ineq_constraints'] = [f'c_1', f'c_2']

        disc_dict[f'{self.ns}.SellarDoeScenario.algo_options'] = {'levels': 'None'
                                                                  }
        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {f'{self.ns}.{self.sc_name}.{self.c_name}.x': 1.,
                       f'{self.ns}.{self.sc_name}.{self.c_name}.y_1': 1.,
                       f'{self.ns}.{self.sc_name}.{self.c_name}.y_2': 1.,
                       f'{self.ns}.{self.sc_name}.{self.c_name}.z': array([
                           1., 1.]), f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv': local_dv}
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.configure()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ {self.sc_name}',
                       f'\t\t|_ {self.c_name}',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1',
                       '\t\t\t|_ Sellar_Problem']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        exec_eng.execute()

        # retrieve discipline to check the result...
        doe_disc = exec_eng.dm.get_disciplines_with_name(
            "doe." + self.sc_name)[0]

        doe_disc_output = doe_disc.get_sosdisc_outputs()
        XY_pd = doe_disc_output['doe_ds_io']
        # print(XY_pd.columns)
        X_pd = XY_pd['design_parameters']
        Y_pd = XY_pd['functions']

        dimension = sum([len(sublist) if isinstance(
            sublist, list) else 1 for sublist in list(self.dspace['value'].values)])
        full_factorial_samples = len(X_pd)

        theoretical_fullfact_levels = int(n_samples ** (1.0 / dimension))

        theoretical_fullfact_samples = theoretical_fullfact_levels ** dimension
        self.assertEqual(full_factorial_samples, theoretical_fullfact_samples)

        my_optim_result = doe_disc_output['optim_result']
        # print(my_optim_result['x_opt'])
        # print(my_optim_result['f_opt'])

    def test_5_doe_scenario_eval_mode(self):
        print("\n Test 05 : Sellar doe with eval_mode")

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        self.repo_discopt = 'sostrades_core.sos_processes.test'
        self.proc_name_discopt = 'test_sellar_doe_discopt'
        builder = factory.get_builder_from_process(repo=self.repo_discopt,
                                                   mod_id=self.proc_name_discopt)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        disc_dict = {}
        # DoE inputs
        disc_dict[f'{self.ns}.SellarDoeScenario.n_samples'] = 10
        # 'lhs', 'fullfact', ...
        disc_dict[f'{self.ns}.SellarDoeScenario.algo'] = "lhs"
        disc_dict[f'{self.ns}.SellarDoeScenario.design_space'] = self.dspace_optim
        disc_dict[f'{self.ns}.SellarDoeScenario.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{self.ns}.SellarDoeScenario.objective_name'] = 'obj'
        # disc_dict[f'{self.ns}.SellarDoeScenario.ineq_constraints'] = [f'c_1', f'c_2']

        disc_dict[f'{self.ns}.SellarDoeScenario.algo_options'] = {'levels': 'None'
                                                                  }
        disc_dict[f'{self.ns}.SellarDoeScenario.eval_mode'] = True

        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        # array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.sub_mda_class'] = 'MDANewtonRaphson'
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = array([2.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = array([2.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = array([2.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.z'] = array([
            2., 2.])

        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        self.assertFalse(exec_eng.dm.get_data(
            f'{self.ns}.SellarDoeScenario.algo_options', 'editable'))
        self.assertFalse(exec_eng.dm.get_data(
            f'{self.ns}.SellarDoeScenario.algo', 'editable'))

        exec_eng.execute()

        # Check that the jacobian has not been executed
        self.assertEqual(
            exec_eng.root_process.sos_disciplines[0].sos_disciplines[0].jac, None)
        # Exec_eng with only the coupling
        exec_eng2 = ExecutionEngine(self.study_name)
        factory = exec_eng2.factory

        repo_discopt = 'sostrades_core.sos_processes.test'
        proc_name_discopt = 'test_sellar_doe_discopt'
        builder = factory.get_builder_from_process(repo=repo_discopt,
                                                   mod_id='test_sellar_coupling')

        factory.set_builders_to_coupling_builder(builder)
        exec_eng2.configure()

        # -- set up disciplines in Scenario
        disc_dict = {}
        # DoE inputs
        disc_dict[f'{self.ns}.SellarDoeScenario.n_samples'] = 10
        # 'lhs', 'fullfact', ...
        disc_dict[f'{self.ns}.SellarDoeScenario.algo'] = "lhs"
        disc_dict[f'{self.ns}.SellarDoeScenario.design_space'] = self.dspace_optim
        disc_dict[f'{self.ns}.SellarDoeScenario.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{self.ns}.SellarDoeScenario.objective_name'] = 'obj'
        # disc_dict[f'{self.ns}.SellarDoeScenario.ineq_constraints'] = [f'c_1', f'c_2']

        disc_dict[f'{self.ns}.SellarDoeScenario.algo_options'] = {'levels': 'None'
                                                                  }
        disc_dict[f'{self.ns}.SellarDoeScenario.eval_mode'] = True

        exec_eng.load_study_from_input_dict(
            disc_dict)  # here it is not exec_eng2

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        # array([1.])
        # Rem: here no more {self.sc_name}
        values_dict[f'{self.ns}.{self.c_name}.sub_mda_class'] = 'MDANewtonRaphson'
        values_dict[f'{self.ns}.{self.c_name}.x'] = array([2.])
        values_dict[f'{self.ns}.{self.c_name}.y_1'] = array([2.])
        values_dict[f'{self.ns}.{self.c_name}.y_2'] = array([2.])
        values_dict[f'{self.ns}.{self.c_name}.z'] = array([2., 2.])

        values_dict[f'{self.ns}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng2.load_study_from_input_dict(values_dict)
        exec_eng2.execute()

    def test_6_doe_scenario_eval_mode_with_eval_jac(self):
        print("\n Test 06 : Sellar doe with eval_mode and eval_jac")

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        repo_discopt = 'sostrades_core.sos_processes.test'
        proc_name_discopt = 'test_sellar_doe_discopt'
        builder = factory.get_builder_from_process(repo=repo_discopt,
                                                   mod_id=proc_name_discopt)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up design space
        dspace_dict = {'variable': ['x', 'z'],
                       'value': [[2.], [2., 2.]],
                       'lower_bnd': [[0.], [-10., 0.]],
                       'upper_bnd': [[10.], [10., 10.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True, True]]}
        #                   'type' : ['float',['float','float'],'float','float']
        dspace = pd.DataFrame(dspace_dict)

        # -- set up disciplines in Scenario
        disc_dict = {}
        # DoE inputs
        disc_dict[f'{self.ns}.SellarDoeScenario.n_samples'] = 10
        # 'lhs', 'fullfact', ...
        disc_dict[f'{self.ns}.SellarDoeScenario.algo'] = "lhs"
        disc_dict[f'{self.ns}.SellarDoeScenario.design_space'] = dspace
        disc_dict[f'{self.ns}.SellarDoeScenario.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{self.ns}.SellarDoeScenario.objective_name'] = 'obj'
        # disc_dict[f'{self.ns}.SellarDoeScenario.ineq_constraints'] = [f'c_1', f'c_2']

        disc_dict[f'{self.ns}.SellarDoeScenario.algo_options'] = {'levels': 'None'
                                                                  }
        disc_dict[f'{self.ns}.SellarDoeScenario.eval_mode'] = True
        disc_dict[f'{self.ns}.SellarDoeScenario.eval_jac'] = True
        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        # array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.sub_mda_class'] = 'MDANewtonRaphson'
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = array([2.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = array([2.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = array([2.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.z'] = array([
            2., 2.])

        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        # values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.linearization_mode'] = 'adjoint'
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        # Get the jacobian of each functions (constraints + objective)
        computed_jac = exec_eng.root_process.sos_disciplines[0].sos_disciplines[0].jac

        self.assertListEqual(list(computed_jac.keys()), [
            f'{self.ns}.{self.sc_name}.{self.c_name}.obj'])

    def _test_7_doe_CustomDoE(self):
        '''
        TEST COMMENTED BECAUSE MDF FORMULATION BUILD A MDACHAIN INSTEAD OF SOSCOUPLING
        '''
        print("\n Test 07 : Sellar doe with Custom algorithm")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        dspace_dict = {'variable': ['x', 'z', 'y_1', 'y_2'],
                       'value': [[1.], [5., 2.], [1.], [1.]],
                       'lower_bnd': [[0.], [-10., 0.], [-100.], [-100.]],
                       'upper_bnd': [[10.], [10., 10.], [100.], [100.]],
                       'enable_variable': [True, True, True, True],
                       'activated_elem': [[True], [True, True], [True], [True]]}
        #                   'type' : ['float',['float','float'],'float','float']
        self.dspace = pd.DataFrame(dspace_dict)

        self.repo = 'sostrades_core.sos_processes.test'
        self.proc_name = 'test_sellar_doe'

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id=self.proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        doe_file = join(dirname(__file__), 'data', 'X_pd.csv')
        # -- set up disciplines in Scenario
        disc_dict = {}
        n_samples = 100
        # DoE inputs
        disc_dict[f'{self.ns}.SellarDoeScenario.n_samples'] = n_samples
        disc_dict[f'{self.ns}.SellarDoeScenario.algo'] = "CustomDOE"
        disc_dict[f'{self.ns}.SellarDoeScenario.design_space'] = self.dspace_optim
        disc_dict[f'{self.ns}.SellarDoeScenario.formulation'] = 'MDF'
        disc_dict[f'{self.ns}.SellarDoeScenario.objective_name'] = 'obj'
        # disc_dict[f'{self.ns}.SellarDoeScenario.ineq_constraints'] = [f'c_1', f'c_2']
        disc_dict[f'{self.ns}.SellarDoeScenario.algo_options'] = {
            # 'samples': X_pd,
            'skiprows': 1,
            'doe_file': doe_file
        }

        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.x'] = 1.
        values_dict[f'{self.ns}.{self.sc_name}.y_1'] = 1.
        values_dict[f'{self.ns}.{self.sc_name}.y_2'] = 1.
        values_dict[f'{self.ns}.{self.sc_name}.z'] = array([1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.configure()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ {self.sc_name}',
                       '\t\t|_ Sellar_Problem',
                       '\t\t|_ Sellar_2',
                       '\t\t|_ Sellar_1']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        exec_eng.execute()

        # retrieve discipline to check the result...
        doe_disc = exec_eng.dm.get_disciplines_with_name(
            "doe." + self.sc_name)[0]

        doe_disc_output = doe_disc.get_sosdisc_outputs()
        XY_pd = doe_disc_output['doe_ds_io']
        X_pd = XY_pd['design_parameters']
        doe_file_df = pd.read_csv(doe_file)

        self.assertEqual(len(X_pd), n_samples)

        self.assertTrue((X_pd['doe.SellarDoeScenario.x'].values.flatten().round(10) ==
                         doe_file_df['doe.SellarDoeScenario.x'].values.round(10)).all())

    def _test_8_test_doe_scenario_df(self):
        '''
        TEST COMMENTED BECAUSE MDF FORMULATION BUILD A MDACHAIN INSTEAD OF SOSCOUPLING
        '''
        print("\n Test 08: DiscAllTypes doe solution")
        self.study_name = 'doe'
        self.ns = f'{self.study_name}'
        self.sc_name = "DiscAllTypesDoeScenario"
        self.c_name = "DiscAllTypesCoupling"
        print(self.sc_name)

        dspace_dict = {'variable': ['z', 'h'],
                       'value': [[1.], [5., 2.]],
                       'lower_bnd': [[0.], [-10., 0.]],
                       'upper_bnd': [[10.], [10., 10.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True, True]]}
        #                   'type' : ['float',['float','float'],'float','float']
        dspace = pd.DataFrame(dspace_dict)

        self.repo = 'sostrades_core.sos_processes.test'
        self.proc_name = 'test_DiscAllTypes_doe'

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id=self.proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        disc_dict = {}
        n_samples = 100
        # Doe inputs
        disc_dict[f'{self.ns}.DiscAllTypesDoeScenario.n_samples'] = n_samples
        # 'lhs', 'CustomDOE', 'fullfact', ...
        disc_dict[f'{self.ns}.DiscAllTypesDoeScenario.algo'] = 'lhs'
        disc_dict[f'{self.ns}.DiscAllTypesDoeScenario.design_space'] = dspace

        disc_dict[f'{self.ns}.DiscAllTypesDoeScenario.formulation'] = 'MDF'
        disc_dict[f'{self.ns}.DiscAllTypesDoeScenario.objective_name'] = 'o'

        exec_eng.load_study_from_input_dict(disc_dict)

        # DiscAllTypes inputs

        h_data = array([0., 0., 0., 0.])
        dict_in_data = {'key0': 0., 'key1': 0.}
        df_in_data = pd.DataFrame(array([[0.0, 1.0, 2.0], [0.1, 1.1, 2.1],
                                         [0.2, 1.2, 2.2], [-9., -8.7, 1e3]]),
                                  columns=['variable', 'c2', 'c3'])
        weather_data = 'cloudy, it is Toulouse ...'
        dict_of_dict_in_data = {'key_A': {'subKey1': 0.1234, 'subKey2': 111.111, 'subKey3': 2036},
                                'key_B': {'subKey1': 1.2345, 'subKey2': 222.222, 'subKey3': 2036}}
        a_df = pd.DataFrame(array([[5., -.05, 5.e5, 5. ** 5], [2.9, 1., 0., -209.1],
                                   [0.7e-5, 2e3 / 3, 17., 3.1416], [-19., -2., -1e3, 6.6]]),
                            columns=['key1', 'key2', 'key3', 'key4'])
        dict_of_df_in_data = {'key_C': a_df,
                              'key_D': a_df * 3.1416}

        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.z'] = 1.
        values_dict['doe.DiscAllTypesDoeScenario.DiscAllTypes.h'] = h_data
        values_dict['doe.DiscAllTypesDoeScenario.DiscAllTypes.dict_in'] = dict_in_data
        values_dict['doe.DiscAllTypesDoeScenario.DiscAllTypes.df_in'] = df_in_data
        values_dict[f'{self.ns}.{self.sc_name}.weather'] = weather_data
        values_dict['doe.DiscAllTypesDoeScenario.DiscAllTypes.dict_of_dict_in'] = dict_of_dict_in_data
        values_dict['doe.DiscAllTypesDoeScenario.DiscAllTypes.dict_of_df_in'] = dict_of_df_in_data

        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.configure()

        exec_eng.execute()

        # retrieve discipline to check the result...
        doe_disc = exec_eng.dm.get_disciplines_with_name(
            "doe." + self.sc_name)[0]

        doe_disc_output = doe_disc.get_sosdisc_outputs()
        XY_pd = doe_disc_output['doe_ds_io']
        X_pd = XY_pd['design_parameters']

        self.assertEqual(len(X_pd), n_samples)


if '__main__' == __name__:
    cls = TestSoSDOEScenario()
    cls.setUp()
    cls.test_5_doe_scenario_eval_mode()
