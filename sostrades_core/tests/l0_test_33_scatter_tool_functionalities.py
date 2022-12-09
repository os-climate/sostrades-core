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
import pandas as pd
import unittest
from time import sleep
from shutil import rmtree
from pathlib import Path
from os.path import join
import os

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from tempfile import gettempdir
from sostrades_core.tools.builder_info.builder_info_functions import get_ns_list_in_builder_list


class TestVerySimpleMultiScenario(unittest.TestCase):
    """
    SoSVerySimpleMultiScenario test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.dirs_to_del = []
        self.namespace = 'MyCase'
        self.study_name = f'{self.namespace}'
        self.repo = 'sostrades_core.sos_processes.test'
        self.base_path = 'sostrades_core.sos_wrapping.test_discs'
        self.exec_eng = ExecutionEngine(self.namespace)
        self.factory = self.exec_eng.factory
        self.root_dir = gettempdir()

    def tearDown(self):

        for dir_to_del in self.dirs_to_del:
            sleep(0.5)
            if Path(dir_to_del).is_dir():
                rmtree(dir_to_del)
        sleep(0.5)


    def _test_get_ns_list_in_builder_list(self):
        builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                             mod_id='test_disc1_scenario')

        mod_list = f'{self.base_path}.disc3_scenario.Disc3'
        disc3_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc3', mod_list)
        builder_list.append(disc3_builder)

        ns_list  = get_ns_list_in_builder_list(builder_list)
        print(ns_list)
        ns_list_th = ['ns_disc3', 'ns_out_disc3', 'ns_ac', 'ns_data_ac']
        self.assertListEqual(list(ns_list.sort()),list(ns_list_th.sort()))

    def test_01_multi_scenario_driver_with_no_scatter_map(self):

        builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                             mod_id='test_disc1_scenario')

        mod_list = f'{self.base_path}.disc3_scenario.Disc3'
        disc3_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc3', mod_list)
        self.exec_eng.ns_manager.add_ns(
            'ns_disc3', f'{self.namespace}.Disc3')
        self.exec_eng.ns_manager.add_ns(
            'ns_out_disc3', self.namespace)

        builder_list.append(disc3_builder)

        multi_scenarios = self.exec_eng.factory.create_driver(
            'multi_scenarios', builder_list)


        self.exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)
        self.exec_eng.configure()

        dict_values = {}
        scenario_df = pd.DataFrame({'selected_scenario': [True, True],
                                    'scenario_name': ['scenario_1',
                                                      'scenario_2']})
        dict_values[f'{self.study_name}.multi_scenarios.scenario_df'] = scenario_df
        dict_values[f'{self.study_name}.multi_scenarios.builder_mode'] = 'multi_instance'
        dict_values[f'{self.study_name}.multi_scenarios.instance_reference'] = False
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        scenario_list = ['scenario_1', 'scenario_2']
        for scenario in scenario_list:
            x1 = 2.
            x2 = 4.
            a1 = 3
            b1 = 4
            a2 = 6
            b2 = 2

            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.a'] = a1
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.x'] = x1
            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc3.constant'] = 3
            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc3.power'] = 2
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_1.Disc1.b'] = b1
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_2.Disc1.b'] = b2
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_1.Disc3.z'] = 1.2
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_2.Disc3.z'] = 1.5


        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        self.exec_eng.execute()
        y1 = a1 * x1 + b1
        y2 = a1 * x1 + b2

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.y'), y1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.y'), y2)

    def test_02_scatter_capabilities(self):

        builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                             mod_id='test_disc1_scenario')

        mod_list = f'{self.base_path}.disc3_scenario.Disc3'
        disc3_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc3', mod_list)
        self.exec_eng.ns_manager.add_ns(
            'ns_disc3', f'{self.namespace}.Disc3')
        self.exec_eng.ns_manager.add_ns(
            'ns_out_disc3', self.namespace)

        builder_list.append(disc3_builder)

        multi_scenarios = self.exec_eng.factory.create_driver(
            'multi_scenarios', builder_list)


        self.exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)
        self.exec_eng.configure()

        dict_values = {}
        scenario_df = pd.DataFrame({'selected_scenario': [True, True],
                                    'scenario_name': ['scenario_1',
                                                      'scenario_2']})
        dict_values[f'{self.study_name}.multi_scenarios.scenario_df'] = scenario_df
        dict_values[f'{self.study_name}.multi_scenarios.builder_mode'] = 'multi_instance'
        dict_values[f'{self.study_name}.multi_scenarios.instance_reference'] = False
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        scenario_list = ['scenario_1', 'scenario_2']
        for scenario in scenario_list:
            x1 = 2.
            x2 = 4.
            a1 = 3
            b1 = 4
            a2 = 6
            b2 = 2

            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.a'] = a1
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.x'] = x1
            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc3.constant'] = 3
            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc3.power'] = 2
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_1.Disc1.b'] = b1
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_2.Disc1.b'] = b2
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_1.Disc3.z'] = 1.2
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_2.Disc3.z'] = 1.5


        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        self.exec_eng.execute()
        y1 = a1 * x1 + b1
        y2 = a1 * x1 + b2

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.y'), y1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.y'), y2)

if '__main__' == __name__:
    cls = TestVerySimpleMultiScenario()
    cls.setUp()
    cls.test_07_scatter_node_namespace_removal_and_change_builder_mode_mono_to_multi()
    cls.tearDown()
