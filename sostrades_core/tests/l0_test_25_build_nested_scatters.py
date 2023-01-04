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
import pandas as pd
from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class TestBuildScatter(unittest.TestCase):
    """
    Scatter build test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.name = 'Coupling'
        self.namespace = 'MyCase'
        self.study_name = f'{self.namespace}'
        self.repo = 'sostrades_core.sos_processes.test'
        self.sub_proc = 'test_disc1_disc2_coupling'
        self.exec_eng = ExecutionEngine(self.namespace)
        self.factory = self.exec_eng.factory

    def test_01_build_coupling_of_scatter(self):
        '''
        Execution treeview is the same but display treeview should look like :
        |_disc
            |_scatter1
            |_scatter2
        '''
        driver_name = 'coupling_scatter'
        builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                             mod_id=self.sub_proc)
        # builder_list is a list of builders from self.sub_proc
        dict_values = {}
        dict_values[f'{self.study_name}.{driver_name}.builder_mode'] = 'multi_instance'
        # User fill in the fields in the GUI

        with self.assertRaises(Exception) as cm:
            self.exec_eng0 = ExecutionEngine(self.namespace)
            scatter_list = self.exec_eng0.factory.create_driver(
                driver_name, builder_list, display_options=['group_scenarios_under_disciplines'])
            self.exec_eng0.factory.set_builders_to_coupling_builder(scatter_list)

            self.exec_eng0.configure()
            self.exec_eng0.load_study_from_input_dict(dict_values)
        error_message = f'The display options parameter for the driver creation should be a dict'
        self.assertEqual(str(cm.exception), error_message)

        with self.assertRaises(Exception) as cm:
            self.exec_eng00 = ExecutionEngine(self.namespace)
            scatter_list = self.exec_eng00.factory.create_driver(
                driver_name, builder_list, display_options={'wrong_option': True})
            self.exec_eng00.factory.set_builders_to_coupling_builder(scatter_list)

            self.exec_eng00.configure()
            self.exec_eng00.load_study_from_input_dict(dict_values)
        error_message = f"Display options should be in the possible list : ['hide_under_coupling', 'hide_coupling_in_driver', 'group_scenarios_under_disciplines']"
        self.assertEqual(str(cm.exception), error_message)

        scatter_list = self.exec_eng.factory.create_driver(
            driver_name, builder_list, display_options={'group_scenarios_under_disciplines': True})

        self.exec_eng.factory.set_builders_to_coupling_builder(scatter_list)

        self.exec_eng.configure()

        scenario_df = pd.DataFrame({'selected_scenario': [True, True],
                                    'scenario_name': ['scatter1',
                                                      'scatter2']})

        dict_values[f'{self.study_name}.{driver_name}.scenario_df'] = scenario_df
        # User fill in the fields in the GUI

        self.exec_eng.load_study_from_input_dict(dict_values)

        self.exec_eng.display_treeview_nodes()

        x1 = 2.
        a1 = 3
        b1 = 4
        b2 = 2
        scatter_list = ['scatter1', 'scatter2']
        for scatter in scatter_list:
            dict_values[f'{self.study_name}.{driver_name}.{scatter}.Disc1.a'] = a1
            dict_values[f'{self.study_name}.{driver_name}.{scatter}.x'] = x1
            dict_values[f'{self.study_name}.{driver_name}.{scatter}.Disc2.constant'] = 3
            dict_values[f'{self.study_name}.{driver_name}.{scatter}.Disc2.power'] = 2
        dict_values[f'{self.study_name}.{driver_name}.scatter1.Disc1.b'] = b1
        dict_values[f'{self.study_name}.{driver_name}.scatter2.Disc1.b'] = b2

        self.exec_eng.load_study_from_input_dict(dict_values)

        # exec treeview
        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       f'\t|_ {driver_name}',
                       f'\t\t|_ scatter1',
                       f'\t\t\t|_ Disc1',
                       f'\t\t\t|_ Disc2',
                       f'\t\t|_ scatter2',
                       f'\t\t\t|_ Disc1',
                       f'\t\t\t|_ Disc2', ]

        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes(exec_display=True)

        # exec treeview
        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       f'\t|_ {driver_name}',
                       f'\t\t|_ Disc1',
                       f'\t\t\t|_ scatter1',
                       f'\t\t\t|_ scatter2',
                       f'\t\t|_ Disc2',
                       f'\t\t\t|_ scatter1',
                       f'\t\t\t|_ scatter2',
                       f'\t\t|_ scatter1',
                       f'\t\t|_ scatter2',
                       ]

        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()
        self.exec_eng.execute()
