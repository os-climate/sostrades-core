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

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine


class TestMultiScenarioSensitivityanalysis(unittest.TestCase):
    """
    MultiScenario and Sensitivityanalysis processes test class
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

    def test_01_multi_scenario_of_SA(self):

        # scatter build map
        ac_map = {'input_name': 'name_list',
                  'input_type': 'string_list',
                  'input_ns': 'ns_scatter_scenario',
                  'output_name': 'ac_name',
                  'scatter_ns': 'ns_ac',
                  'gather_ns': 'ns_scenario',
                  'ns_to_update': ['ns_data_ac']}

        self.exec_eng.smaps_manager.add_build_map('name_list', ac_map)

        # scenario build map
        scenario_map = {'input_name': 'scenario_list',
                        'input_type': 'string_list',
                        'input_ns': 'ns_scatter_scenario',
                        'output_name': 'scenario_name',
                        'scatter_ns': 'ns_scenario',
                        'gather_ns': 'ns_scatter_scenario',
                        'ns_to_update': ['ns_disc3', 'ns_barrierr', 'ns_out_disc3']}

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

        # instantiate factory # get instantiator from Discipline class

        builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                             mod_id='test_disc1_scenario')

        scatter_list = self.exec_eng.factory.create_multi_scatter_builder_from_list(
            'name_list', builder_list=builder_list, autogather=False)

        mod_list = f'{self.base_path}.disc3_scenario.Disc3'
        disc3_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc3', mod_list)
        scatter_list.append(disc3_builder)

        sa_builder = self.exec_eng.factory.create_evaluator_builder(
            'SA', 'sensitivity', scatter_list)

        multi_scenarios = self.exec_eng.factory.create_multi_scenario_builder(
            'multi_scenarios', 'scenario_list', [sa_builder], autogather=False)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)

        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()

        x1 = 2
        x2 = 4

        dict_values = {f'{self.study_name}.multi_scenarios.name_1.x_trade': [x1, x2],
                       f'{self.study_name}.multi_scenarios.trade_variables': {'name_1.x': 'float'}}

        self.exec_eng.load_study_from_input_dict(dict_values)

        self.exec_eng.display_treeview_nodes()

        dict_values = {self.study_name +
                       '.multi_scenarios.name_list': ['name_1', 'name_2']}

        self.exec_eng.load_study_from_input_dict(dict_values)

        self.exec_eng.display_treeview_nodes()

        private_val = {}
        scenario_list = ['scenario_1', 'scenario_2']
        for scenario in scenario_list:
            a1 = 3
            b1 = 4
            a2 = 6
            b2 = 2
            x2b = 5.0

            private_val[self.study_name + '.name_1.a'] = a1
            private_val[self.study_name + '.name_2.a'] = a2
            private_val[self.study_name + '.name_2.x'] = x2b
            private_val[self.study_name + '.multi_scenarios.' +
                        scenario + '.SA.Disc1.name_1.b'] = b1
            private_val[self.study_name + '.multi_scenarios.' +
                        scenario + '.SA.Disc1.name_2.b'] = b2
            private_val[self.study_name + '.multi_scenarios.' +
                        scenario + '.SA.Disc3.constant'] = 3
            private_val[self.study_name + '.multi_scenarios.' +
                        scenario + '.SA.Disc3.power'] = 2
        private_val[self.study_name +
                    '.multi_scenarios.scenario_1.Disc3.z'] = 1.2
        private_val[self.study_name +
                    '.multi_scenarios.scenario_2.Disc3.z'] = 1.5

        private_val[self.study_name +
                    '.multi_scenarios.scenario_1.SA.eval_inputs'] = ['z']
        private_val[self.study_name +
                    '.multi_scenarios.scenario_2.SA.eval_inputs'] = ['z']
        private_val[self.study_name +
                    '.multi_scenarios.scenario_1.SA.eval_outputs'] = ['o']
        private_val[self.study_name +
                    '.multi_scenarios.scenario_2.SA.eval_outputs'] = ['o']

        self.exec_eng.load_study_from_input_dict(private_val)

        self.exec_eng.execute()

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.name_1.x'), x1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.name_1.x'), x2)

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.SA.name_1.y'), a1 * x1 + b1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.SA.name_2.y'), a2 * x2b + b2)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.SA.name_1.y'), a1 * x2 + b1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.SA.name_2.y'), a2 * x2b + b2)

    def test_02_SA_of_multi_scenario(self):

        builder_list = self.exec_eng.factory.get_builder_from_process(repo=self.repo,
                                                                      mod_id='test_disc1_disc3_multi_scenario')

        sa_builder = self.exec_eng.factory.create_evaluator_builder(
            'SA', 'sensitivity', builder_list)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            sa_builder)

        for ns in self.exec_eng.ns_manager.ns_list:
            self.exec_eng.ns_manager.update_namespace_with_extra_ns(
                ns, 'SA', after_name=self.exec_eng.study_name)

        self.exec_eng.configure()

        self.exec_eng.display_treeview_nodes()

        x1 = 2
        x2 = 4

        dict_values = {f'{self.study_name}.SA.multi_scenarios.trade_variables': {'name_1.x': 'float'},
                       f'{self.study_name}.SA.multi_scenarios.name_1.x_trade': [x1, x2],
                       f'{self.study_name}.SA.multi_scenarios.name_list': ['name_1', 'name_2']}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        private_val = {}
        scenario_list = ['scenario_1', 'scenario_2']
        for scenario in scenario_list:
            a1 = 3
            b1 = 4
            a2 = 6
            b2 = 2
            x2b = 5

            private_val[self.study_name + '.SA.name_1.a'] = a1
            private_val[self.study_name + '.SA.name_2.a'] = a2
            private_val[self.study_name + '.SA.name_2.x'] = x2b
            private_val[self.study_name + '.SA.multi_scenarios.' +
                        scenario + '.Disc1.name_1.b'] = b1
            private_val[self.study_name + '.SA.multi_scenarios.' +
                        scenario + '.Disc1.name_2.b'] = b2
            private_val[self.study_name + '.SA.multi_scenarios.' +
                        scenario + '.Disc3.constant'] = 3
            private_val[self.study_name + '.SA.multi_scenarios.' +
                        scenario + '.Disc3.power'] = 2
        private_val[self.study_name +
                    '.SA.multi_scenarios.scenario_1.Disc3.z'] = 1.2
        private_val[self.study_name +
                    '.SA.multi_scenarios.scenario_2.Disc3.z'] = 1.5

        private_val[self.study_name +
                    '.SA.eval_inputs'] = ['z']
        private_val[self.study_name +
                    '.SA.eval_inputs'] = ['z']
        private_val[self.study_name +
                    '.SA.eval_outputs'] = ['o']
        private_val[self.study_name +
                    '.SA.eval_outputs'] = ['o']

        self.exec_eng.load_study_from_input_dict(private_val)

        self.exec_eng.execute()

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.SA.multi_scenarios.scenario_1.name_1.x'), x1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.SA.multi_scenarios.scenario_2.name_1.x'), x2)

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.SA.multi_scenarios.scenario_1.name_1.y'), a1 * x1 + b1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.SA.multi_scenarios.scenario_1.name_2.y'), a2 * x2b + b2)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.SA.multi_scenarios.scenario_2.name_1.y'), a1 * x2 + b1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.SA.multi_scenarios.scenario_2.name_2.y'), a2 * x2b + b2)
