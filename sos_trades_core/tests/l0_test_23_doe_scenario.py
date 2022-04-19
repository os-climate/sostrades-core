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
from time import time

from gemseo.algos.doe.doe_factory import DOEFactory
from pandas._testing import assert_frame_equal

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

        dspace_dict_eval = {'variable': ['x', 'z'],
                            'lower_bnd': [[0.], [-10., 0.]],
                            'upper_bnd': [[10.], [10., 10.]]
                            }

        self.dspace = pd.DataFrame(dspace_dict)
        self.dspace_eval = pd.DataFrame(dspace_dict_eval)
        self.dspace_optim = pd.DataFrame(dspace_dict_optim)

        input_selection_local_dv_x = {'selected_input': [True, True, False, False, False],
                                      'full_name': ['DoEEval.Sellar_Problem.local_dv', 'x', 'y_1',
                                                    'y_2',
                                                    'z']}
        self.input_selection_local_dv_x = pd.DataFrame(
            input_selection_local_dv_x)

        input_selection_x_z = {'selected_input': [False, True, False, False, True],
                               'full_name': ['DoEEval.Sellar_Problem.local_dv', 'x', 'y_1',
                                             'y_2',
                                             'z']}
        self.input_selection_x_z = pd.DataFrame(input_selection_x_z)

        input_selection_x = {'selected_input': [False, True, False, False, False],
                             'full_name': ['DoEEval.Sellar_Problem.local_dv', 'x', 'y_1',
                                           'y_2',
                                           'z']}
        self.input_selection_x = pd.DataFrame(input_selection_x)

        input_selection_local_dv = {'selected_input': [True, False, False, False, False],
                                    'full_name': ['DoEEval.Sellar_Problem.local_dv', 'x', 'y_1',
                                                  'y_2',
                                                  'z']}
        self.input_selection_local_dv = pd.DataFrame(input_selection_local_dv)

        output_selection_obj = {'selected_output': [False, False, True, False, False],
                                'full_name': ['c_1', 'c_2', 'obj', 'y_1', 'y_2']}
        self.output_selection_obj = pd.DataFrame(output_selection_obj)

        output_selection_obj_y1_y2 = {'selected_output': [False, False, True, True, True],
                                      'full_name': ['c_1', 'c_2', 'obj', 'y_1', 'y_2']}
        self.output_selection_obj_y1_y2 = pd.DataFrame(
            output_selection_obj_y1_y2)

        self.repo = 'sos_trades_core.sos_processes.test'
        self.proc_name = 'test_sellar_doe'

    def test_1_doe_scenario_check_treeview(self):
        print("\n Test 1 : check configure and treeview")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id=self.proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # print('\n in test doe scenario')
        # for key in exec_eng.dm.data_id_map:
        #     print("key", key)

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

        repo_discopt = 'sos_trades_core.sos_processes.test'
        proc_name_discopt = 'test_sellar_doe_discopt'
        builder = factory.get_builder_from_process(repo=repo_discopt,
                                                   mod_id=proc_name_discopt)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # print('\n in test doe scenario')
        # for key in exec_eng.dm.data_id_map:
        #     print("key", key)

        # -- set up disciplines in Scenario
        disc_dict = {}
        # DoE inputs
        n_samples = 100
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
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = 1.
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = 1.
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = 1.
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

            repo_discopt = 'sos_trades_core.sos_processes.test'
            proc_name_discopt = 'test_sellar_doe_discopt'
            builder = factory.get_builder_from_process(repo=repo_discopt,
                                                       mod_id=proc_name_discopt)

            exec_eng.factory.set_builders_to_coupling_builder(builder)

            exec_eng.configure()
            n_samples = 100
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
            values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = 1.
            values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = 1.
            values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = 1.
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

    def test_4_test_options_full_fact(self):
        print("\n Test 04: Sellar doe solution check with DisciplinaryOpt formulation/ fullfact algo")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        repo_discopt = 'sos_trades_core.sos_processes.test'
        proc_name_discopt = 'test_sellar_doe_discopt'
        builder = factory.get_builder_from_process(repo=repo_discopt,
                                                   mod_id=proc_name_discopt)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()
        n_samples = 100
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
        values_dict = {}
        # array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = 1.
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = 1.
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = 1.
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

        self.repo_discopt = 'sos_trades_core.sos_processes.test'
        self.proc_name_discopt = 'test_sellar_doe_discopt'
        builder = factory.get_builder_from_process(repo=self.repo_discopt,
                                                   mod_id=self.proc_name_discopt)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        disc_dict = {}
        # DoE inputs
        disc_dict[f'{self.ns}.SellarDoeScenario.n_samples'] = 100
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
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = 2.
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = 2.
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = 2.
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

        repo_discopt = 'sos_trades_core.sos_processes.test'
        proc_name_discopt = 'test_sellar_doe_discopt'
        builder = factory.get_builder_from_process(repo=repo_discopt,
                                                   mod_id='test_sellar_coupling')

        factory.set_builders_to_coupling_builder(builder)
        exec_eng2.configure()

        # -- set up disciplines in Scenario
        disc_dict = {}
        # DoE inputs
        disc_dict[f'{self.ns}.SellarDoeScenario.n_samples'] = 100
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

        repo_discopt = 'sos_trades_core.sos_processes.test'
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
        disc_dict[f'{self.ns}.SellarDoeScenario.n_samples'] = 100
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

        self.repo = 'sos_trades_core.sos_processes.test'
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
        # print(XY_pd)
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

        self.repo = 'sos_trades_core.sos_processes.test'
        self.proc_name = 'test_DiscAllTypes_doe'

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id=self.proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # print('\n in test doe scenario')
        # for key in exec_eng.dm.data_id_map:
        #     print("key", key)

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
        # disc_dict[f'{self.ns}.DiscAllTypesDoeScenario.ineq_constraints'] = [f'c_1', f'c_2']
        # disc_dict[f'{self.ns}.DiscAllTypesDoeScenario.algo_options'] = {'levels': 'None'}
        #

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
        # print(XY_pd)
        X_pd = XY_pd['design_parameters']

        self.assertEqual(len(X_pd), n_samples)

    def test_9_usepydoe_lib(self):

        pydoe_lib = DOEFactory()
        for algo_name in pydoe_lib.algorithms:
            if algo_name not in ['ccdesign', 'CustomDOE', 'DiagonalDOE', 'OT_FACTORIAL', 'OT_COMPOSITE', 'OT_AXIAL',
                                 'OT_OPT_LHS']:
                print(algo_name)
                algo = pydoe_lib.create(algo_name)
                samples = algo._generate_samples(
                    n_samples=100, dimension=5)
                # print(samples)

    def test_10_execute_all_algos(self):
        print("\n Test 04: Sellar doe solution check with DisciplinaryOpt formulation/ fullfact algo")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        repo_discopt = 'sos_trades_core.sos_processes.test'
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

        disc_dict[f'{self.ns}.SellarDoeScenario.design_space'] = self.dspace_optim
        disc_dict[f'{self.ns}.SellarDoeScenario.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{self.ns}.SellarDoeScenario.objective_name'] = 'obj'
        # disc_dict[f'{self.ns}.SellarDoeScenario.ineq_constraints'] = [f'c_1', f'c_2']
        doe_file = join(dirname(__file__), 'data', 'X_pd.csv')
        disc_dict[f'{self.ns}.SellarDoeScenario.algo_options'] = {
            'levels': 'None'}
        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        # array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = 1.
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = 1.
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = 1.
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.z'] = array([
            1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)
        available_algorithms = exec_eng.root_process.sos_disciplines[0].get_available_driver_names(
        )
        print(available_algorithms)
        for algo in available_algorithms:
            # need more options for these algorithms
            if algo not in ['CustomDOE', 'DiagonalDOE', 'OT_FACTORIAL', 'OT_COMPOSITE', 'OT_AXIAL']:
                print(algo)
                disc_dict[f'{self.ns}.SellarDoeScenario.algo'] = algo
                exec_eng.load_study_from_input_dict(disc_dict)
                exec_eng.execute()

    def test_11_doe_eval_execution_fullfact(self):

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = "test_sellar_doe_eval"
        doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        disc_dict = {}
        # DoE inputs
        n_samples = 100
        disc_dict[f'{self.ns}.DoEEval.sampling_algo'] = "fullfact"
        disc_dict[f'{self.ns}.DoEEval.design_space'] = self.dspace_eval
        disc_dict[f'{self.ns}.DoEEval.algo_options'] = {'n_samples': n_samples, 'fake_option': 'fake_option'}
        disc_dict[f'{self.ns}.DoEEval.eval_inputs'] = self.input_selection_x_z
        disc_dict[f'{self.ns}.DoEEval.eval_outputs'] = self.output_selection_obj_y1_y2
        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        # array([1.])
        values_dict[f'{self.ns}.x'] = array([1.])
        values_dict[f'{self.ns}.y_1'] = array([1.])
        values_dict[f'{self.ns}.y_2'] = array([1.])
        values_dict[f'{self.ns}.z'] = array([1., 1.])
        values_dict[f'{self.ns}.DoEEval.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ DoEEval',
                       '\t\t|_ Sellar_2',
                       '\t\t|_ Sellar_1',
                       '\t\t|_ Sellar_Problem']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()
        doe_disc = exec_eng.dm.get_disciplines_with_name('doe.DoEEval')[0]

        doe_disc_samples = doe_disc.get_sosdisc_outputs(
            'doe_samples_dataframe')

        dimension = sum([len(sublist) if isinstance(
            sublist, list) else 1 for sublist in list(self.dspace_eval['lower_bnd'].values)])

        theoretical_fullfact_levels = int(n_samples ** (1.0 / dimension))

        theoretical_fullfact_samples = theoretical_fullfact_levels ** dimension
        self.assertEqual(len(doe_disc_samples), theoretical_fullfact_samples)

    def test_12_doe_eval_CustomDoE(self):

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = "test_sellar_doe_eval"
        doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        disc_dict = {}
        # DoE inputs

        disc_dict[f'{self.ns}.DoEEval.sampling_algo'] = "CustomDOE"

        disc_dict[f'{self.ns}.DoEEval.eval_inputs'] = self.input_selection_x_z
        disc_dict[f'{self.ns}.DoEEval.eval_outputs'] = self.output_selection_obj_y1_y2

        x_values = [array([9.379763880395856]), array([8.88644794300546]),
                    array([3.7137135749628882]), array([0.0417022004702574]), array([6.954954792150857])]
        z_values = [array([1.515949043849158, 5.6317362409322165]),
                    array([-1.1962705421254114, 6.523436208612142]),
                    array([-1.9947578026244557, 4.822570933860785]
                          ), array([1.7490668861813, 3.617234050834533]),
                    array([-9.316161097119341, 9.918161285133076])]

        samples_dict = {'x': x_values, 'z': z_values}
        samples_df = pd.DataFrame(samples_dict)
        disc_dict[f'{self.ns}.DoEEval.custom_samples_df'] = samples_df

        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        # array([1.])
        values_dict[f'{self.ns}.x'] = array([1.])
        values_dict[f'{self.ns}.y_1'] = array([1.])
        values_dict[f'{self.ns}.y_2'] = array([1.])
        values_dict[f'{self.ns}.z'] = array([1., 1.])
        values_dict[f'{self.ns}.DoEEval.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ DoEEval',
                       '\t\t|_ Sellar_2',
                       '\t\t|_ Sellar_1',
                       '\t\t|_ Sellar_Problem']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()
        doe_disc = exec_eng.dm.get_disciplines_with_name('doe.DoEEval')[0]

        doe_disc_samples = doe_disc.get_sosdisc_outputs(
            'doe_samples_dataframe')
        doe_disc_obj = doe_disc.get_sosdisc_outputs('obj_dict')
        doe_disc_y1 = doe_disc.get_sosdisc_outputs('y_1_dict')
        doe_disc_y2 = doe_disc.get_sosdisc_outputs('y_2_dict')
        self.assertEqual(len(doe_disc_samples), 5)
        self.assertEqual(len(doe_disc_obj), 5)
        self.assertEqual(len(doe_disc_y1), 5)
        self.assertEqual(len(doe_disc_y2), 5)

    def test_13_doe_eval_execution_lhs_on_1_var(self):

        dspace_dict_x = {'variable': ['x'],

                         'lower_bnd': [0.],
                         'upper_bnd': [10.],

                         }
        dspace_x = pd.DataFrame(dspace_dict_x)

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = "test_sellar_doe_eval"
        doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        disc_dict = {}
        # DoE inputs
        n_samples = 10
        disc_dict[f'{self.ns}.DoEEval.sampling_algo'] = "lhs"
        disc_dict[f'{self.ns}.DoEEval.design_space'] = dspace_x
        disc_dict[f'{self.ns}.DoEEval.algo_options'] = {'n_samples': n_samples}
        disc_dict[f'{self.ns}.DoEEval.eval_inputs'] = self.input_selection_x
        disc_dict[f'{self.ns}.DoEEval.eval_outputs'] = self.output_selection_obj_y1_y2
        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        # array([1.])
        values_dict[f'{self.ns}.x'] = array([1.])
        values_dict[f'{self.ns}.y_1'] = array([1.])
        values_dict[f'{self.ns}.y_2'] = array([1.])
        values_dict[f'{self.ns}.z'] = array([1., 1.])
        values_dict[f'{self.ns}.DoEEval.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ DoEEval',
                       '\t\t|_ Sellar_2',
                       '\t\t|_ Sellar_1',
                       '\t\t|_ Sellar_Problem']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()
        doe_disc = exec_eng.dm.get_disciplines_with_name('doe.DoEEval')[0]

        doe_disc_samples = doe_disc.get_sosdisc_outputs(
            'doe_samples_dataframe')
        self.assertEqual(len(doe_disc_samples), n_samples)

    def test_14_doe_eval_options_and_design_space_after_reconfiguration(self):

        default_algo_options_lhs = {
            'n_samples': 'default',
            'alpha': 'orthogonal',
            'eval_jac': False,
            'face': 'faced',
            'iterations': 5,
            'max_time': 0,
            'seed': 1,
            'center_bb': 'default',
            'center_cc': 'default',
            'criterion': 'default',
            'levels': 'default'
        }

        dspace_dict_x = {'variable': ['x'],

                         'lower_bnd': [[0.]],
                         'upper_bnd': [[10.]]
                         }
        dspace_x = pd.DataFrame(dspace_dict_x)

        dspace_dict_x_eval = {'variable': ['x'],

                              'lower_bnd': [[5.]],
                              'upper_bnd': [[11.]]
                              }
        dspace_x_eval = pd.DataFrame(dspace_dict_x_eval)

        dspace_dict_x_local_dv = {'variable': ['x', 'DoEEval.Sellar_Problem.local_dv'],

                                  'lower_bnd': [[0.], 0.],
                                  'upper_bnd': [[10.], 10.]
                                  }
        dspace_x_local_dv = pd.DataFrame(dspace_dict_x_local_dv)

        dspace_dict_x_z = {'variable': ['x', 'z'],

                           'lower_bnd': [[0.], [0., 0.]],
                           'upper_bnd': [[10.], [10., 10.]]
                           }
        dspace_x_z = pd.DataFrame(dspace_dict_x_z)

        dspace_dict_eval = {'variable': ['x', 'z'],

                            'lower_bnd': [[0.], [-10., 0.]],
                            'upper_bnd': [[10.], [10., 10.]]
                            }
        dspace_eval = pd.DataFrame(dspace_dict_eval)

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = "test_sellar_doe_eval"
        doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ DoEEval',
                       '\t\t|_ Sellar_2',
                       '\t\t|_ Sellar_1',
                       '\t\t|_ Sellar_Problem']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        # -- set up disciplines
        values_dict = {}
        values_dict[f'{self.ns}.x'] = array([1.])
        values_dict[f'{self.ns}.y_1'] = array([1.])
        values_dict[f'{self.ns}.y_2'] = array([1.])
        values_dict[f'{self.ns}.z'] = array([1., 1.])
        values_dict[f'{self.ns}.DoEEval.Sellar_Problem.local_dv'] = 10
        exec_eng.load_study_from_input_dict(values_dict)

        # configure disciplines with the algo lhs
        disc_dict = {}
        disc_dict[f'{self.ns}.DoEEval.sampling_algo'] = "lhs"
        disc_dict[f'{self.ns}.DoEEval.eval_inputs'] = self.input_selection_x
        disc_dict[f'{self.ns}.DoEEval.eval_outputs'] = self.output_selection_obj
        exec_eng.load_study_from_input_dict(disc_dict)
        self.assertDictEqual(exec_eng.dm.get_value(
            'doe.DoEEval.algo_options'), default_algo_options_lhs)
        # WARNING: default design space with array is built with 2-elements arrays : [0., 0.]
        # but dspace_x contains 1-element arrays
#         assert_frame_equal(exec_eng.dm.get_value('doe.DoEEval.design_space').reset_index(drop=True),
#                            dspace_x.reset_index(drop=True), check_dtype=False)

        # trigger a reconfiguration after options and design space changes
        disc_dict = {'doe.DoEEval.algo_options': {'n_samples': 10, 'face': 'faced'},
                     'doe.DoEEval.design_space': dspace_x_eval}
        exec_eng.load_study_from_input_dict(disc_dict)
        self.assertDictEqual(exec_eng.dm.get_value('doe.DoEEval.algo_options'), {
                             'n_samples': 10, 'face': 'faced'})
        assert_frame_equal(exec_eng.dm.get_value('doe.DoEEval.design_space').reset_index(drop=True),
                           dspace_x_eval.reset_index(drop=True), check_dtype=False)

        # trigger a reconfiguration after algo name change
        disc_dict = {'doe.DoEEval.sampling_algo': "fullfact"}
        exec_eng.load_study_from_input_dict(disc_dict)
        self.assertDictEqual(exec_eng.dm.get_value(
            'doe.DoEEval.algo_options'), default_algo_options_lhs)
        assert_frame_equal(exec_eng.dm.get_value('doe.DoEEval.design_space').reset_index(drop=True),
                           dspace_x_eval.reset_index(drop=True), check_dtype=False)

        disc_dict = {'doe.DoEEval.algo_options': {
            'n_samples': 10, 'face': 'faced'}}
        exec_eng.load_study_from_input_dict(disc_dict)
        self.assertDictEqual(exec_eng.dm.get_value('doe.DoEEval.algo_options'), {
                             'n_samples': 10, 'face': 'faced'})

        # trigger a reconfiguration after eval_inputs and eval_outputs changes
        disc_dict = {'doe.DoEEval.eval_outputs': self.output_selection_obj_y1_y2,
                     'doe.DoEEval.eval_inputs': self.input_selection_x_z}
        exec_eng.load_study_from_input_dict(disc_dict)
        self.assertDictEqual(exec_eng.dm.get_value('doe.DoEEval.algo_options'), {
                             'n_samples': 10, 'face': 'faced'})
        # WARNING: default design space with array is built with 2-elements arrays : [0., 0.]
        # but dspace_x contains 1-element arrays
#         assert_frame_equal(exec_eng.dm.get_value('doe.DoEEval.design_space').reset_index(drop=True),
#                            dspace_x_z.reset_index(drop=True), check_dtype=False)
        disc_dict = {'doe.DoEEval.algo_options': {'n_samples': 100, 'face': 'faced'},
                     'doe.DoEEval.eval_outputs': self.output_selection_obj_y1_y2,
                     'doe.DoEEval.design_space': dspace_eval}
        exec_eng.load_study_from_input_dict(disc_dict)
        self.assertDictEqual(exec_eng.dm.get_value('doe.DoEEval.algo_options'),
                             {'n_samples': 100, 'face': 'faced'})
        assert_frame_equal(exec_eng.dm.get_value('doe.DoEEval.design_space').reset_index(drop=True),
                           dspace_eval.reset_index(drop=True), check_dtype=False)

        exec_eng.execute()

    def test_15_doe_eval_CustomDoE_reconfiguration(self):

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = "test_sellar_doe_eval"
        doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ DoEEval',
                       '\t\t|_ Sellar_2',
                       '\t\t|_ Sellar_1',
                       '\t\t|_ Sellar_Problem']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        # -- set up disciplines in Scenario
        disc_dict = {}
        # DoE inputs
        disc_dict[f'{self.ns}.DoEEval.sampling_algo'] = "CustomDOE"
        disc_dict[f'{self.ns}.DoEEval.eval_inputs'] = self.input_selection_x
        disc_dict[f'{self.ns}.DoEEval.eval_outputs'] = self.output_selection_obj
        exec_eng.load_study_from_input_dict(disc_dict)
        self.assertListEqual(exec_eng.dm.get_value(
            'doe.DoEEval.custom_samples_df').columns.tolist(), ['x'])
        disc_dict[f'{self.ns}.DoEEval.eval_inputs'] = self.input_selection_local_dv_x
        exec_eng.load_study_from_input_dict(disc_dict)
        self.assertListEqual(exec_eng.dm.get_value('doe.DoEEval.custom_samples_df').columns.tolist(),
                             ['DoEEval.Sellar_Problem.local_dv', 'x'])
        disc_dict[f'{self.ns}.DoEEval.eval_inputs'] = self.input_selection_local_dv
        exec_eng.load_study_from_input_dict(disc_dict)
        self.assertListEqual(exec_eng.dm.get_value('doe.DoEEval.custom_samples_df').columns.tolist(),
                             ['DoEEval.Sellar_Problem.local_dv'])
        disc_dict[f'{self.ns}.DoEEval.eval_outputs'] = self.output_selection_obj_y1_y2
        disc_dict[f'{self.ns}.DoEEval.eval_inputs'] = self.input_selection_x_z
        exec_eng.load_study_from_input_dict(disc_dict)

        x_values = [array([9.379763880395856]), array([8.88644794300546]),
                    array([3.7137135749628882]), array([0.0417022004702574]), array([6.954954792150857])]
        z_values = [array([1.515949043849158, 5.6317362409322165]),
                    array([-1.1962705421254114, 6.523436208612142]),
                    array([-1.9947578026244557, 4.822570933860785]
                          ), array([1.7490668861813, 3.617234050834533]),
                    array([-9.316161097119341, 9.918161285133076])]

        samples_dict = {'x': x_values, 'z': z_values}
        samples_df = pd.DataFrame(samples_dict)
        disc_dict[f'{self.ns}.DoEEval.custom_samples_df'] = samples_df

        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        # array([1.])
        values_dict[f'{self.ns}.x'] = array([1.])
        values_dict[f'{self.ns}.y_1'] = array([1.])
        values_dict[f'{self.ns}.y_2'] = array([1.])
        values_dict[f'{self.ns}.z'] = array([1., 1.])
        values_dict[f'{self.ns}.DoEEval.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        doe_disc = exec_eng.dm.get_disciplines_with_name('doe.DoEEval')[0]

        doe_disc_samples = doe_disc.get_sosdisc_outputs(
            'doe_samples_dataframe')
        self.assertEqual(len(doe_disc_samples), 5)

    def test_16_doe_eval_design_space_normalisation(self):

        dspace_dict_x_eval = {'variable': ['x'],

                              'lower_bnd': [5.],
                              'upper_bnd': [11.]
                              }
        dspace_x_eval = pd.DataFrame(dspace_dict_x_eval)

        dspace_dict_eval = {'variable': ['x', 'z'],

                            'lower_bnd': [[-9.], [-10., 4.]],
                            'upper_bnd': [[150.], [10., 100.]]
                            }
        dspace_eval = pd.DataFrame(dspace_dict_eval)

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = "test_sellar_doe_eval"
        doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ DoEEval',
                       '\t\t|_ Sellar_2',
                       '\t\t|_ Sellar_1',
                       '\t\t|_ Sellar_Problem']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        # -- set up disciplines
        values_dict = {}
        values_dict[f'{self.ns}.x'] = array([1.])
        values_dict[f'{self.ns}.y_1'] = array([1.])
        values_dict[f'{self.ns}.y_2'] = array([1.])
        values_dict[f'{self.ns}.z'] = array([1., 1.])
        values_dict[f'{self.ns}.DoEEval.Sellar_Problem.local_dv'] = 10
        exec_eng.load_study_from_input_dict(values_dict)

        # configure disciplines with the algo lhs and check that generated
        # samples are within default bounds
        disc_dict = {}
        disc_dict[f'{self.ns}.DoEEval.sampling_algo'] = "lhs"
        disc_dict[f'{self.ns}.DoEEval.eval_inputs'] = self.input_selection_x
        disc_dict[f'{self.ns}.DoEEval.eval_outputs'] = self.output_selection_obj_y1_y2

        disc_dict['doe.DoEEval.algo_options'] = {'n_samples': 10, 'face': 'faced'}
        exec_eng.load_study_from_input_dict(disc_dict)
        exec_eng.execute()
        # check that all generated samples are within [0,10.] range
        generated_x = exec_eng.dm.get_value(
            'doe.DoEEval.doe_samples_dataframe')['x'].tolist()
        self.assertTrue(all(0 <= element[0] <= 10. for element in generated_x))

        # trigger a reconfiguration after options and design space changes
        disc_dict = {'doe.DoEEval.design_space': dspace_x_eval}
        exec_eng.load_study_from_input_dict(disc_dict)
        exec_eng.execute()
        # check that all generated samples are within [5.,11.] range
        generated_x = exec_eng.dm.get_value(
            'doe.DoEEval.doe_samples_dataframe')['x'].tolist()
        self.assertTrue(all(5. <= element[0] <= 11. for element in generated_x))

        # trigger a reconfiguration after algo name change
        disc_dict = {'doe.DoEEval.sampling_algo': "fullfact",
                     'doe.DoEEval.eval_outputs': self.output_selection_obj_y1_y2,
                     'doe.DoEEval.eval_inputs': self.input_selection_x_z,
                     'doe.DoEEval.design_space': dspace_eval}

        exec_eng.load_study_from_input_dict(disc_dict)
        disc_dict['doe.DoEEval.algo_options'] = {'n_samples': 10, 'face': 'faced'}
        exec_eng.load_study_from_input_dict(disc_dict)
        exec_eng.execute()
        generated_x = exec_eng.dm.get_value(
            'doe.DoEEval.doe_samples_dataframe')['x'].tolist()
        self.assertTrue(all(-9. <= element[0] <= 150. for element in generated_x))

        generated_z = exec_eng.dm.get_value(
            'doe.DoEEval.doe_samples_dataframe')['z'].tolist()
        self.assertTrue(
            all(-10. <= element[0] <= 10. and 4. <= element[1] <= 100. for element in
                generated_z))

    def test_17_doe_eval_CustomDoE_reconfiguration_after_execution(self):

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = "test_sellar_doe_eval"
        doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        disc_dict = {}
        # DoE inputs

        disc_dict[f'{self.ns}.DoEEval.sampling_algo'] = "CustomDOE"

        disc_dict[f'{self.ns}.DoEEval.eval_inputs'] = self.input_selection_local_dv_x
        disc_dict[f'{self.ns}.DoEEval.eval_outputs'] = self.output_selection_obj_y1_y2

        x_values = [array([9.379763880395856]), array([8.88644794300546]),
                    array([3.7137135749628882]), array([0.0417022004702574]), array([6.954954792150857])]
        local_dv_values = x_values

        samples_dict = {'x': x_values,
                        'DoEEval.Sellar_Problem.local_dv': local_dv_values}
        samples_df = pd.DataFrame(samples_dict)
        disc_dict[f'{self.ns}.DoEEval.custom_samples_df'] = samples_df

        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        # array([1.])
        values_dict[f'{self.ns}.x'] = array([1.])
        values_dict[f'{self.ns}.y_1'] = array([1.])
        values_dict[f'{self.ns}.y_2'] = array([1.])
        values_dict[f'{self.ns}.z'] = array([1., 1.])
        values_dict[f'{self.ns}.DoEEval.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ DoEEval',
                       '\t\t|_ Sellar_2',
                       '\t\t|_ Sellar_1',
                       '\t\t|_ Sellar_Problem']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()
        doe_disc = exec_eng.dm.get_disciplines_with_name('doe.DoEEval')[0]

        doe_disc_samples = doe_disc.get_sosdisc_outputs(
            'doe_samples_dataframe')
        doe_disc_obj = doe_disc.get_sosdisc_outputs('obj_dict')
        doe_disc_y1 = doe_disc.get_sosdisc_outputs('y_1_dict')
        doe_disc_y2 = doe_disc.get_sosdisc_outputs('y_2_dict')
        self.assertEqual(len(doe_disc_samples), 5)
        self.assertEqual(len(doe_disc_obj), 5)
        self.assertEqual(len(doe_disc_y1), 5)
        self.assertEqual(len(doe_disc_y2), 5)

        disc_dict = {f'{self.ns}.DoEEval.eval_inputs': self.input_selection_x}
        exec_eng.load_study_from_input_dict(disc_dict)
        self.assertEqual(len(doe_disc_samples), 5)
        self.assertEqual(len(doe_disc_obj), 5)
        self.assertEqual(len(doe_disc_y1), 5)
        self.assertEqual(len(doe_disc_y2), 5)

    def _test_18_doe_eval_parallel_execution_time(self):
        execution_time = 0
        for i in range(5):
            start = time()

            dspace_dict_x = {'variable': ['x'],

                             'lower_bnd': [0.],
                             'upper_bnd': [1000.],

                             }
            dspace_x = pd.DataFrame(dspace_dict_x)

            exec_eng = ExecutionEngine(self.study_name)
            factory = exec_eng.factory

            proc_name = "test_sellar_doe_eval"
            doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                                mod_id=proc_name)

            exec_eng.factory.set_builders_to_coupling_builder(
                doe_eval_builder)

            exec_eng.configure()

            # -- set up disciplines in Scenario
            disc_dict = {}
            # DoE inputs
            n_samples = 1000
            disc_dict[f'{self.ns}.DoEEval.sampling_algo'] = "lhs"
            disc_dict[f'{self.ns}.DoEEval.design_space'] = dspace_x
            disc_dict[f'{self.ns}.DoEEval.algo_options'] = {'n_samples': n_samples}
            disc_dict[f'{self.ns}.DoEEval.eval_inputs'] = self.input_selection_x
            disc_dict[f'{self.ns}.DoEEval.eval_outputs'] = self.output_selection_obj_y1_y2
            exec_eng.load_study_from_input_dict(disc_dict)

            # Sellar inputs
            local_dv = 10.
            values_dict = {}
            # array([1.])
            values_dict[f'{self.ns}.x'] = array([1.])
            values_dict[f'{self.ns}.y_1'] = array([1.])
            values_dict[f'{self.ns}.y_2'] = array([1.])
            values_dict[f'{self.ns}.z'] = array([1., 1.])
            values_dict[f'{self.ns}.DoEEval.Sellar_Problem.local_dv'] = local_dv
            exec_eng.load_study_from_input_dict(values_dict)

            exec_eng.execute()
            stop = time()

            print(str(stop - start))
            execution_time += stop - start
        print("sequential execution in " + str(execution_time / 5) + " seconds")

    def _test_19_doe_eval_parallel_execution_time_8_cores(self):
        execution_time = 0
        execution_time_20 = 0
        for i in range(5):
            start = time()

            dspace_dict_x = {'variable': ['x'],

                             'lower_bnd': [0.],
                             'upper_bnd': [1000.],

                             }
            dspace_x = pd.DataFrame(dspace_dict_x)

            exec_eng = ExecutionEngine(self.study_name)
            factory = exec_eng.factory

            proc_name = "test_sellar_doe_eval"
            doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                                mod_id=proc_name)

            exec_eng.factory.set_builders_to_coupling_builder(
                doe_eval_builder)

            exec_eng.configure()

            # -- set up disciplines in Scenario
            disc_dict = {}
            # DoE inputs
            n_samples = 1000
            disc_dict[f'{self.ns}.DoEEval.sampling_algo'] = "lhs"
            disc_dict[f'{self.ns}.DoEEval.design_space'] = dspace_x
            disc_dict[f'{self.ns}.DoEEval.algo_options'] = {'n_samples': n_samples}
            disc_dict[f'{self.ns}.DoEEval.eval_inputs'] = self.input_selection_x
            disc_dict[f'{self.ns}.DoEEval.eval_outputs'] = self.output_selection_obj_y1_y2
            exec_eng.load_study_from_input_dict(disc_dict)

            # Sellar inputs
            local_dv = 10.
            values_dict = {}
            # array([1.])
            values_dict[f'{self.ns}.x'] = array([1.])
            values_dict[f'{self.ns}.y_1'] = array([1.])
            values_dict[f'{self.ns}.y_2'] = array([1.])
            values_dict[f'{self.ns}.z'] = array([1., 1.])
            values_dict[f'{self.ns}.DoEEval.Sellar_Problem.local_dv'] = local_dv
            exec_eng.load_study_from_input_dict(values_dict)

            exec_eng.execute()
            stop = time()
            # print(str(stop - start))
            execution_time += stop - start

            exec_eng.load_study_from_input_dict({f'{self.ns}.DoEEval.algo_options': {'n_samples': n_samples}})
            start = time()
            exec_eng.execute()
            stop = time()
            # print(str(stop - start))
            execution_time_20 += stop - start

        print("parallel execution with 10 cores in " + 
              str(execution_time / 5) + " seconds")
        print("parallel execution with 20 cores in " + 
              str(execution_time_20 / 5) + " seconds")

    def test_20_doe_eval_with_2_outputs_with_the_same_name(self):

        dspace_dict = {'variable': ['x', 'DoEEval.Disc1.a'],

                       'lower_bnd': [0., 50.],
                       'upper_bnd': [100., 200.],

                       }
        dspace = pd.DataFrame(dspace_dict)

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = "test_disc1_disc2_doe_eval"
        doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ doe',
                       f'\t|_ DoEEval',
                       '\t\t|_ Disc1',
                       '\t\t|_ Disc2']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        # -- set up disciplines
        private_values = {
            self.study_name + '.x': 10.,
            self.study_name + '.DoEEval.Disc1.a': 5.,
            self.study_name + '.DoEEval.Disc1.b': 25431.,
            self.study_name + '.y': 4.,
            self.study_name + '.DoEEval.Disc2.constant': 3.1416,
            self.study_name + '.DoEEval.Disc2.power': 2}
        exec_eng.load_study_from_input_dict(private_values)

        # configure disciplines with the algo lhs and check that generated
        # samples are within default bounds

        input_selection_x_a = {'selected_input': [True, True],
                               'full_name': ['x', 'DoEEval.Disc1.a']}
        input_selection_x_a = pd.DataFrame(input_selection_x_a)

        output_selection_z_z = {'selected_output': [True, True],
                                'full_name': ['z', 'DoEEval.Disc1.z']}
        output_selection_z_z = pd.DataFrame(output_selection_z_z)

        disc_dict = {}
        disc_dict[f'{self.ns}.DoEEval.sampling_algo'] = "lhs"
        disc_dict[f'{self.ns}.DoEEval.eval_inputs'] = input_selection_x_a
        disc_dict[f'{self.ns}.DoEEval.eval_outputs'] = output_selection_z_z

        exec_eng.load_study_from_input_dict(disc_dict)
        disc_dict = {'doe.DoEEval.algo_options': {'n_samples': 100, 'face': 'faced'}, 'doe.DoEEval.design_space': dspace}

        exec_eng.load_study_from_input_dict(disc_dict)
        exec_eng.execute()
        self.assertEqual(len(exec_eng.dm.get_value(
            'doe.DoEEval.Disc1.z_dict')), 100)
        self.assertEqual(len(exec_eng.dm.get_value('doe.z_dict')), 100)


if '__main__' == __name__:
    cls = TestSoSDOEScenario()
    cls.setUp()
    cls.test_14_doe_eval_options_and_design_space_after_reconfiguration()
