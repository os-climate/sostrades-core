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
from pandas._testing import assert_frame_equal
import pprint
import numpy as np
import pandas as pd
from time import sleep
from shutil import rmtree
from pathlib import Path
from os.path import join

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from copy import deepcopy
from tempfile import gettempdir


class TestMultiScenarioOfDoeEval(unittest.TestCase):
    """
    MultiScenario and doe_eval processes test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.namespace = 'MyCase'
        self.study_name = f'{self.namespace}'
        self.repo = 'sos_trades_core.sos_processes.test'
        self.base_path = 'sos_trades_core.sos_wrapping.test_discs'
        self.exec_eng = ExecutionEngine(self.namespace)
        self.factory = self.exec_eng.factory

    def setup_my_usecase(self):
        '''
        '''
        ######### Numerical values   ####
        x_1 = 2
        x_2_a = 4
        x_2_b = 5

        a_1 = 3
        b_1 = 4
        a_2 = 6
        b_2 = 2

        constant = 3
        power = 2
        z_1 =  1.2
        z_2 =  1.5

        my_doe_algo = "lhs"
        n_samples = 4


        ######### Selection of variables and DS  ####
        input_selection_z_scenario_1 = {
            'selected_input': [False, False, False, True, False, False],
            'full_name': ['x', 'a','b','multi_scenarios.scenario_1.Disc3.z','constant','power']}
        input_selection_z_scenario_1 = pd.DataFrame(input_selection_z_scenario_1)

        input_selection_z_scenario_2 = {
            'selected_input': [False, False, False, True, False, False],
            'full_name': ['x', 'a','b','multi_scenarios.scenario_2.Disc3.z','constant','power']}
        input_selection_z_scenario_2 = pd.DataFrame(input_selection_z_scenario_2)

        output_selection_o_scenario_1 = {
            'selected_output': [False, False, True],
            'full_name': ['indicator', 'y', 'multi_scenarios.scenario_1.o']}
        output_selection_o_scenario_1 = pd.DataFrame(output_selection_o_scenario_1)

        output_selection_o_scenario_2 = {
            'selected_output': [False, False, True],
            'full_name': ['indicator', 'y', 'multi_scenarios.scenario_2.o']}
        output_selection_o_scenario_2 = pd.DataFrame(output_selection_o_scenario_2)

        dspace_dict_z = {'variable': ['z'],
                          'lower_bnd': [0.],
                          'upper_bnd': [10.],
                          'enable_variable': [True],
                          'activated_elem': [[True]]}
        dspace_z = pd.DataFrame(dspace_dict_z)

        my_name_list = ['name_1', 'name_2']

        my_x_trade = [x_1, x_2_a]

        my_trade_variables = {'name_1.x': 'float'}

        ######### Fill the dictionary for dm   ####
        dict_values = {}

        prefix = f'{self.study_name}.multi_scenarios'

        dict_values[f'{self.study_name}.name_2.x'] = x_2_b
        dict_values[f'{self.study_name}.name_1.a'] = a_1
        dict_values[f'{self.study_name}.name_2.a'] = a_2

        dict_values[f'{prefix}.name_1.x_trade'] = my_x_trade
        dict_values[f'{prefix}.trade_variables'] = my_trade_variables

        dict_values[f'{prefix}.name_list'] =  my_name_list

        dict_values[ f'{prefix}.scenario_1.DoE_Eval.Disc1.name_1.b'] = b_1
        dict_values[ f'{prefix}.scenario_1.DoE_Eval.Disc1.name_2.b'] = b_2

        dict_values[ f'{prefix}.scenario_2.DoE_Eval.Disc1.name_1.b'] = b_1
        dict_values[ f'{prefix}.scenario_2.DoE_Eval.Disc1.name_2.b'] = b_2

        dict_values[f'{prefix}.scenario_1.DoE_Eval.Disc3.constant'] = constant
        dict_values[f'{prefix}.scenario_1.DoE_Eval.Disc3.power'] = power
        dict_values[f'{prefix}.scenario_1.Disc3.z'] = z_1 # reference value (computed in any case)

        dict_values[f'{prefix}.scenario_2.DoE_Eval.Disc3.constant'] = constant
        dict_values[f'{prefix}.scenario_2.DoE_Eval.Disc3.power'] = power
        dict_values[f'{prefix}.scenario_2.Disc3.z'] = z_2 # reference value (computed in any case)

        dict_values[f'{prefix}.scenario_1.DoE_Eval.sampling_algo'] = my_doe_algo
        dict_values[f'{prefix}.scenario_1.DoE_Eval.eval_inputs'] = input_selection_z_scenario_1
        dict_values[f'{prefix}.scenario_1.DoE_Eval.eval_outputs'] = output_selection_o_scenario_1


        dict_values[f'{prefix}.scenario_2.DoE_Eval.sampling_algo'] = my_doe_algo
        dict_values[f'{prefix}.scenario_2.DoE_Eval.eval_inputs'] = input_selection_z_scenario_2
        dict_values[f'{prefix}.scenario_2.DoE_Eval.eval_outputs'] = output_selection_o_scenario_2

        dict_values[f'{prefix}.scenario_1.DoE_Eval.design_space'] = dspace_z
        dict_values[f'{prefix}.scenario_1.DoE_Eval.algo_options'] = {'n_samples': n_samples}
        dict_values[f'{prefix}.scenario_2.DoE_Eval.design_space'] = dspace_z
        dict_values[f'{prefix}.scenario_2.DoE_Eval.algo_options'] = {'n_samples': n_samples}

        return dict_values


    def test_01_multi_scenario_of_doe_eval(self):
        '''
        '''
        # scatter build map
        ac_map = {'input_name': 'name_list',

                  'input_ns': 'ns_scatter_scenario',
                  'output_name': 'ac_name',
                  'scatter_ns': 'ns_ac',
                  'gather_ns': 'ns_scenario',
                  'ns_to_update': ['ns_data_ac']}

        self.exec_eng.smaps_manager.add_build_map('name_list', ac_map)

        # scenario build map
        scenario_map = {'input_name': 'scenario_list',

                        'input_ns': 'ns_scatter_scenario',
                        'output_name': 'scenario_name',
                        'scatter_ns': 'ns_scenario',
                        'gather_ns': 'ns_scatter_scenario',
                        'ns_to_update': ['ns_disc3', 'ns_doe_eval', 'ns_barrierr', 'ns_out_disc3']}

        self.exec_eng.smaps_manager.add_build_map(
            'scenario_list', scenario_map)

        # shared namespace
        self.exec_eng.ns_manager.add_ns('ns_barrierr', 'MyCase')
        self.exec_eng.ns_manager.add_ns(
            'ns_scatter_scenario', 'MyCase.multi_scenarios')
        self.exec_eng.ns_manager.add_ns(
            'ns_disc3', 'MyCase.multi_scenarios.Disc3')
        self.exec_eng.ns_manager.add_ns(
            'ns_out_disc3', 'MyCase.multi_scenarios')
        self.exec_eng.ns_manager.add_ns(
            'ns_data_ac', 'MyCase')
        self.exec_eng.ns_manager.add_ns(
            'ns_doe_eval', f'{self.exec_eng.study_name}.multi_scenarios.DoE_Eval')

        # instantiate factory # get instantiator from Discipline class

        builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                             mod_id='test_disc1_scenario')

        scatter_list = self.exec_eng.factory.create_multi_scatter_builder_from_list(
            'name_list', builder_list=builder_list, autogather=False)

        mod_list = f'{self.base_path}.disc3_scenario.Disc3'
        disc3_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc3', mod_list)
        scatter_list.append(disc3_builder)

        doe_eval_builder = self.exec_eng.factory.create_evaluator_builder(
            'DoE_Eval', 'doe_eval', scatter_list)

        multi_scenarios = self.exec_eng.factory.create_multi_scenario_builder(
            'multi_scenarios', 'scenario_list', [doe_eval_builder], autogather=False)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)

        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()

        dict_values = self.setup_my_usecase()

        # Provide inputs and reconfigure
        self.exec_eng.load_study_from_input_dict(dict_values)

        self.exec_eng.display_treeview_nodes()

        self.exec_eng.display_treeview_nodes(True)

        # Run
        self.exec_eng.execute()

        # Check results
        prefix = f'{self.study_name}.multi_scenarios'
        x_2_b = dict_values[f'{self.study_name}.name_2.x']
        a_1 = dict_values[f'{self.study_name}.name_1.a']
        a_2 = dict_values[f'{self.study_name}.name_2.a']
        my_x_trade = dict_values[f'{prefix}.name_1.x_trade']
        x_1 = my_x_trade[0]
        x_2_a = my_x_trade[1]
        b_1 = dict_values[ f'{prefix}.scenario_1.DoE_Eval.Disc1.name_1.b']
        b_2 = dict_values[ f'{prefix}.scenario_1.DoE_Eval.Disc1.name_2.b']
        z_1 = dict_values[f'{prefix}.scenario_1.Disc3.z']
        z_2 = dict_values[f'{prefix}.scenario_2.Disc3.z']
        constant = dict_values[f'{prefix}.scenario_2.DoE_Eval.Disc3.constant']
        power = dict_values[f'{prefix}.scenario_2.DoE_Eval.Disc3.power']

        self.assertEqual(self.exec_eng.dm.get_value(
                                         f'{prefix}.scenario_1.name_1.x'), x_1)
        self.assertEqual(self.exec_eng.dm.get_value(
                                        f'{prefix}.scenario_2.name_1.x'),x_2_a)
        self.assertEqual(self.exec_eng.dm.get_value(
                         f'{prefix}.scenario_1.DoE_Eval.name_1.y'),a_1 * x_1 + b_1)
        self.assertEqual(self.exec_eng.dm.get_value(
                         f'{prefix}.scenario_1.DoE_Eval.name_2.y'),a_2 * x_2_b + b_2)
        self.assertEqual(self.exec_eng.dm.get_value(
                         f'{prefix}.scenario_2.DoE_Eval.name_1.y'),a_1 * x_2_a + b_1)
        self.assertEqual(self.exec_eng.dm.get_value(
                         f'{prefix}.scenario_2.DoE_Eval.name_2.y'),a_2 * x_2_b + b_2)



    def test_02_multi_scenario_of_doe_eval_from_process(self):
        '''
        '''
        # load process in GUI
        builders = self.factory.get_builder_from_process(
            repo=self.repo, mod_id='test_multiscenario_of_doe_eval')
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()

        dict_values = self.setup_my_usecase()

        # Provide inputs and reconfigure
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()
        self.exec_eng.display_treeview_nodes(True)

        # Run
        self.exec_eng.execute()
