'''
Copyright 2022 Airbus SAS
Modifications on 2023/10/03-2023/11/03 Copyright 2023 Capgemini

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

        self.scenario_list = ['scenario_1', 'scenario_2']
        self.x1 = 2.
        self.a1 = 3
        self.b1 = 4
        self.b2 = 2
        self.y1 = self.a1 * self.x1 + self.b1
        self.y2 = self.a1 * self.x1 + self.b2

        self.builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                                  mod_id='test_disc1_scenario')

        mod_list = f'{self.base_path}.disc3_scenario.Disc3'
        disc3_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc3', mod_list)
        self.exec_eng.ns_manager.add_ns(
            'ns_disc3', f'{self.namespace}.Disc3')
        self.exec_eng.ns_manager.add_ns(
            'ns_out_disc3', self.namespace)

        self.builder_list.append(disc3_builder)

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

        ns_list = get_ns_list_in_builder_list(builder_list)
        print(ns_list)
        ns_list_th = ['ns_ac', 'ns_data_ac', 'ns_disc3', 'ns_out_disc3']
        self.assertListEqual(list(ns_list.sort()), ns_list_th)

    def test_01_multi_scenario_driver_with_no_scatter_map(self):

        multi_scenarios = self.exec_eng.factory.create_multi_instance_driver('multi_scenarios', self.builder_list)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)
        self.exec_eng.configure()

        dict_values = {}
        samples_df = pd.DataFrame({'selected_scenario': [True, True],
                                   'scenario_name': self.scenario_list})
        dict_values[f'{self.study_name}.multi_scenarios.samples_df'] = samples_df
        dict_values[f'{self.study_name}.multi_scenarios.instance_reference'] = False
        self.exec_eng.load_study_from_input_dict(dict_values)

        for scenario in self.scenario_list:
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.a'] = self.a1
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.x'] = self.x1
            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc3.constant'] = 3
            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc3.power'] = 2
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_1.Disc1.b'] = self.b1
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_2.Disc1.b'] = self.b2
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_1.Disc3.z'] = 1.2
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_2.Disc3.z'] = 1.5

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.execute()
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.y'), self.y1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.y'), self.y2)

    def test_02_ns_to_update_capability(self):
        '''
        Check that only namespaces in ns_to_update list are updated
        '''

        self.exec_eng.scattermap_manager.add_build_map('new_map'
                                                       , {'ns_to_update': ['ns_ac', 'ns_out_disc3']})

        multi_scenarios = self.exec_eng.factory.create_multi_instance_driver('multi_scenarios', self.builder_list,
                                                                             map_name='new_map')

        self.exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)
        self.exec_eng.configure()

        dict_values = {}
        samples_df = pd.DataFrame({'selected_scenario': [True, True],
                                   'scenario_name': self.scenario_list})
        dict_values[f'{self.study_name}.multi_scenarios.samples_df'] = samples_df
        dict_values[f'{self.study_name}.multi_scenarios.instance_reference'] = False
        self.exec_eng.load_study_from_input_dict(dict_values)

        # ns_data_ac has not been updated
        self.assertListEqual(self.exec_eng.dm.get_all_namespaces_from_var_name('x'), ['MyCase.x'])
        # ns_disc3 has not been updated
        self.assertListEqual(self.exec_eng.dm.get_all_namespaces_from_var_name('z'), ['MyCase.Disc3.z'])
        # ns_ac has  been updated
        self.assertListEqual(self.exec_eng.dm.get_all_namespaces_from_var_name('y'),
                             ['MyCase.multi_scenarios.scenario_1.y',
                              'MyCase.multi_scenarios.scenario_2.y']
                             )
        # ns_out_disc3 has been updated
        self.assertListEqual(self.exec_eng.dm.get_all_namespaces_from_var_name('o'),
                             ['MyCase.multi_scenarios.scenario_1.o', 'MyCase.multi_scenarios.scenario_2.o']
                             )

        dict_values[f'{self.study_name}.a'] = self.a1
        dict_values[f'{self.study_name}.x'] = self.x1
        dict_values[f'{self.study_name}.Disc3.z'] = 1.5
        scenario_list = ['scenario_1', 'scenario_2']
        for scenario in scenario_list:
            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc3.constant'] = 3
            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc3.power'] = 2
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_1.Disc1.b'] = self.b1
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_2.Disc1.b'] = self.b2

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.execute()

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.y'), self.y1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.y'), self.y2)

    def test_03_ns_not_to_update_capability(self):
        '''
        Check that only namespaces in ns_to_update list are updated
        '''

        with self.assertRaises(Exception) as cm:
            self.exec_eng.scattermap_manager.add_build_map('new_map'
                                                           , {'ns_to_update': ['ns_ac', 'ns_out_disc3'],
                                                              'ns_not_to_update': ['ns_data_ac', 'ns_disc3']})

        error_message = 'The scatter map new_map can not have both ns_to_update and ns_not_to_update keys'
        self.assertEqual(str(cm.exception), error_message)

        self.exec_eng.scattermap_manager.add_build_map('new_map'
                                                       , {'ns_not_to_update': ['ns_data_ac', 'ns_disc3']})

        multi_scenarios = self.exec_eng.factory.create_multi_instance_driver('multi_scenarios', self.builder_list,
                                                                             map_name='new_map')

        self.exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)
        self.exec_eng.configure()

        dict_values = {}
        samples_df = pd.DataFrame({'selected_scenario': [True, True],
                                   'scenario_name': self.scenario_list})
        dict_values[f'{self.study_name}.multi_scenarios.samples_df'] = samples_df
        dict_values[f'{self.study_name}.multi_scenarios.instance_reference'] = False
        self.exec_eng.load_study_from_input_dict(dict_values)

        # ns_data_ac has not been updated
        self.assertListEqual(self.exec_eng.dm.get_all_namespaces_from_var_name('x'), ['MyCase.x'])
        # ns_disc3 has not been updated
        self.assertListEqual(self.exec_eng.dm.get_all_namespaces_from_var_name('z'), ['MyCase.Disc3.z'])
        # ns_ac has  been updated
        self.assertListEqual(self.exec_eng.dm.get_all_namespaces_from_var_name('y'),
                             ['MyCase.multi_scenarios.scenario_1.y',
                              'MyCase.multi_scenarios.scenario_2.y']
                             )
        # ns_out_disc3 has been updated
        self.assertListEqual(self.exec_eng.dm.get_all_namespaces_from_var_name('o'),
                             ['MyCase.multi_scenarios.scenario_1.o', 'MyCase.multi_scenarios.scenario_2.o']
                             )

        dict_values[f'{self.study_name}.a'] = self.a1
        dict_values[f'{self.study_name}.x'] = self.x1
        dict_values[f'{self.study_name}.Disc3.z'] = 1.5
        scenario_list = ['scenario_1', 'scenario_2']
        for scenario in scenario_list:
            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc3.constant'] = 3
            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc3.power'] = 2
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_1.Disc1.b'] = self.b1
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_2.Disc1.b'] = self.b2

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.execute()

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.y'), self.y1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.y'), self.y2)

    def test_04_scatter_list_output_capability(self):
        '''
        Check that the scatter_list is correctly created in output of the driver
        '''
        scatter_list_name = 'scenario_list'
        self.exec_eng.scattermap_manager.add_build_map('new_map', {'scatter_list': (scatter_list_name, 'ns_list')})
        ns_list_value = f'{self.study_name}.multi_scenarios'
        self.exec_eng.ns_manager.add_ns('ns_list', ns_list_value)
        multi_scenarios = self.exec_eng.factory.create_multi_instance_driver('multi_scenarios', self.builder_list,
                                                                             map_name='new_map')

        self.exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)
        self.exec_eng.configure()

        dict_values = {}
        samples_df = pd.DataFrame({'selected_scenario': [True, True],
                                   'scenario_name': self.scenario_list})
        dict_values[f'{self.study_name}.multi_scenarios.samples_df'] = samples_df
        dict_values[f'{self.study_name}.multi_scenarios.instance_reference'] = False
        self.exec_eng.load_study_from_input_dict(dict_values)

        scenario_list_full_name = self.exec_eng.dm.get_all_namespaces_from_var_name('scenario_list')[0]

        self.assertEqual(scenario_list_full_name, f'{ns_list_value}.{scatter_list_name}')

        scenario_list_value = self.exec_eng.dm.get_value(scenario_list_full_name)

        self.assertEqual(scenario_list_value, self.scenario_list)

    def test_05_scatter_name_output_capability(self):
        '''
        Check that the scatter_list is correctly created in output of the driver
        '''
        scatter_name = 'scenario_name'
        self.exec_eng.scattermap_manager.add_build_map('new_map', {'scatter_name': scatter_name})

        multi_scenarios = self.exec_eng.factory.create_multi_instance_driver('multi_scenarios', self.builder_list,
                                                                             map_name='new_map')

        self.exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)
        self.exec_eng.configure()

        dict_values = {}
        samples_df = pd.DataFrame({'selected_scenario': [True, True],
                                   'scenario_name': self.scenario_list})
        dict_values[f'{self.study_name}.multi_scenarios.samples_df'] = samples_df
        dict_values[f'{self.study_name}.multi_scenarios.instance_reference'] = False
        self.exec_eng.load_study_from_input_dict(dict_values)

        scenario_name_list = self.exec_eng.dm.get_all_namespaces_from_var_name(scatter_name)

        self.assertEqual(scenario_name_list,
                         [f'{self.study_name}.multi_scenarios.{scenario}.{scatter_name}' for scenario in
                          self.scenario_list])

        for scenario in self.scenario_list:
            scenario_name_value = self.exec_eng.dm.get_value(
                f'{self.study_name}.multi_scenarios.{scenario}.{scatter_name}')

            self.assertEqual(scenario_name_value, scenario)

    def test_06_multi_scenario_driver_with_dynamic_new_namespace(self):

        dynamic_builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                                     mod_id='test_disc10_dynamic')
        multi_scenarios = self.exec_eng.factory.create_multi_instance_driver('multi_scenarios', dynamic_builder_list)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)
        self.exec_eng.configure()

        dict_values = {}
        samples_df = pd.DataFrame({'selected_scenario': [True, True],
                                   'scenario_name': self.scenario_list})
        dict_values[f'{self.study_name}.multi_scenarios.samples_df'] = samples_df
        dict_values[f'{self.study_name}.multi_scenarios.instance_reference'] = False
        self.exec_eng.load_study_from_input_dict(dict_values)

        for scenario in self.scenario_list:
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.Disc10.Model_Type'] = 'Affine'
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.Disc10.x'] = self.x1
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.Disc10.b'] = 2
        self.exec_eng.load_study_from_input_dict(dict_values)

        b_list = self.exec_eng.dm.get_all_namespaces_from_var_name('b')

        self.assertEqual(b_list,
                         [f'{self.study_name}.multi_scenarios.{scenario}.Disc10.b' for scenario in self.scenario_list])

        self.exec_eng.execute()


if '__main__' == __name__:
    cls = TestVerySimpleMultiScenario()
    cls.setUp()
    cls.test_02_ns_to_update_capability()
    cls.tearDown()
