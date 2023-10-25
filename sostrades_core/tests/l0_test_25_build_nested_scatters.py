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
from sostrades_core.execution_engine.gather_discipline import GatherDiscipline


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

        # builder_list is a list of builders from self.sub_proc
        dict_values = {}
        # User fill in the fields in the GUI

        with self.assertRaises(Exception) as cm:
            self.exec_eng0 = ExecutionEngine(self.namespace)
            self.factory = self.exec_eng0.factory
            builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                                 mod_id=self.sub_proc)
            scatter_list = self.exec_eng0.factory.create_multi_instance_driver(driver_name, builder_list)
            self.exec_eng0.factory.set_builders_to_coupling_builder(scatter_list)

            self.exec_eng0.configure()
            dict_values[f'{self.study_name}.{driver_name}.display_options'] = ['group_scenarios_under_disciplines']
            self.exec_eng0.load_study_from_input_dict(dict_values)
        error_message = f'The display options parameter for the driver creation should be a dict'
        self.assertEqual(str(cm.exception), error_message)

        with self.assertRaises(Exception) as cm:
            self.exec_eng00 = ExecutionEngine(self.namespace)
            self.factory = self.exec_eng00.factory
            builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                                 mod_id=self.sub_proc)
            scatter_list = self.exec_eng00.factory.create_multi_instance_driver(driver_name, builder_list)
            self.exec_eng00.factory.set_builders_to_coupling_builder(scatter_list)

            self.exec_eng00.configure()
            dict_values[f'{self.study_name}.{driver_name}.display_options'] = {'wrong_option': True}
            self.exec_eng00.load_study_from_input_dict(dict_values)
        error_message = f"Display options should be in the possible list : ['hide_under_coupling', 'hide_coupling_in_driver', 'group_scenarios_under_disciplines', 'autogather']"
        self.assertEqual(str(cm.exception), error_message)

        self.factory = self.exec_eng.factory
        builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                             mod_id=self.sub_proc)
        scatter_list = self.exec_eng.factory.create_multi_instance_driver(driver_name, builder_list)

        self.exec_eng.factory.set_builders_to_coupling_builder(scatter_list)

        self.exec_eng.configure()

        dict_values[f'{self.study_name}.{driver_name}.display_options'] = {
            'group_scenarios_under_disciplines': True}
        samples_df = pd.DataFrame({'selected_scenario': [True, True],
                                   'scenario_name': ['scatter1',
                                                     'scatter2']})

        dict_values[f'{self.study_name}.{driver_name}.samples_df'] = samples_df
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

    def test_02_autogather_with_coupling_of_scatter(self):
        '''
        Test autogather when group_scenarios_under_disciplines is activated
        One gather per discipline
        '''
        driver_name = 'coupling_scatter'
        builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                             mod_id=self.sub_proc)
        # builder_list is a list of builders from self.sub_proc
        dict_values = {}
        # User fill in the fields in the GUI

        scatter_list = self.exec_eng.factory.create_multi_instance_driver(driver_name, builder_list, )

        self.exec_eng.factory.set_builders_to_coupling_builder(scatter_list)

        self.exec_eng.configure()

        samples_df = pd.DataFrame({'selected_scenario': [True, True],
                                   'scenario_name': ['scatter1',
                                                     'scatter2']})
        dict_values[f'{self.study_name}.{driver_name}.display_options'] = {
            'group_scenarios_under_disciplines': True,
            'autogather': True}
        dict_values[f'{self.study_name}.{driver_name}.samples_df'] = samples_df
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

        self.assertTrue(isinstance(
            self.exec_eng.dm.get_disciplines_with_name('MyCase.coupling_scatter.Disc1')[0].mdo_discipline_wrapp.wrapper,
            GatherDiscipline))
        self.assertTrue(isinstance(
            self.exec_eng.dm.get_disciplines_with_name('MyCase.coupling_scatter.Disc2')[0].mdo_discipline_wrapp.wrapper,
            GatherDiscipline))
        # exec treeview
        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       f'\t|_ {driver_name}',
                       f'\t\t|_ Disc1',
                       f'\t\t|_ Disc2',
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

        y_gather = self.exec_eng.dm.get_value('MyCase.coupling_scatter.Disc1.y_gather')
        y_gather_th = {scatter_name: self.exec_eng.dm.get_value(
            f'{self.study_name}.{driver_name}.{scatter_name}.y') for scatter_name in scatter_list}

        self.assertDictEqual(y_gather, y_gather_th)

        indicator_gather = self.exec_eng.dm.get_value('MyCase.coupling_scatter.Disc1.indicator_gather')
        indicator_gather_th = {scatter_name: self.exec_eng.dm.get_value(
            f'{self.study_name}.{driver_name}.{scatter_name}.Disc1.indicator') for scatter_name in scatter_list}

        self.assertDictEqual(indicator_gather, indicator_gather_th)

        z_gather = self.exec_eng.dm.get_value('MyCase.coupling_scatter.Disc2.z_gather')
        z_gather_th = {scatter_name: self.exec_eng.dm.get_value(
            f'{self.study_name}.{driver_name}.{scatter_name}.z') for scatter_name in scatter_list}

        self.assertDictEqual(z_gather, z_gather_th)

    def test_03_autogather_without_display_option(self):
        '''
        Test autogather when group_scenarios_under_disciplines is not activated
        One gather at driver node
        '''
        driver_name = 'coupling_scatter'
        builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                             mod_id=self.sub_proc)
        # builder_list is a list of builders from self.sub_proc
        dict_values = {}
        # User fill in the fields in the GUI

        scatter_list = self.exec_eng.factory.create_multi_instance_driver(driver_name, builder_list)

        self.exec_eng.factory.set_builders_to_coupling_builder(scatter_list)

        self.exec_eng.configure()

        samples_df = pd.DataFrame({'selected_scenario': [True, True],
                                   'scenario_name': ['scatter1',
                                                     'scatter2']})
        dict_values[f'{self.study_name}.{driver_name}.display_options'] = {
            'group_scenarios_under_disciplines': False,
            'autogather': True}
        dict_values[f'{self.study_name}.{driver_name}.samples_df'] = samples_df
        # User fill in the fields in the GUI

        self.exec_eng.load_study_from_input_dict(dict_values)

        self.assertTrue(isinstance(
            self.exec_eng.dm.get_disciplines_with_name('MyCase.coupling_scatter_gather')[
                0].mdo_discipline_wrapp.wrapper,
            GatherDiscipline))

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
                       f'\t\t\t|_ Disc2',
                       f'\t|_ {driver_name}_gather',
                       ]

        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes(exec_display=True)

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
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()
        self.exec_eng.execute()

        y_gather = self.exec_eng.dm.get_value('MyCase.coupling_scatter_gather.y_gather')
        y_gather_th = {scatter_name: self.exec_eng.dm.get_value(
            f'{self.study_name}.{driver_name}.{scatter_name}.y') for scatter_name in scatter_list}

        self.assertDictEqual(y_gather, y_gather_th)

        indicator_gather = self.exec_eng.dm.get_value('MyCase.coupling_scatter_gather.indicator_gather')
        indicator_gather_th = {f'{scatter_name}.Disc1': self.exec_eng.dm.get_value(
            f'{self.study_name}.{driver_name}.{scatter_name}.Disc1.indicator') for scatter_name in scatter_list}

        self.assertDictEqual(indicator_gather, indicator_gather_th)

        z_gather = self.exec_eng.dm.get_value('MyCase.coupling_scatter_gather.z_gather')
        z_gather_th = {scatter_name: self.exec_eng.dm.get_value(
            f'{self.study_name}.{driver_name}.{scatter_name}.z') for scatter_name in scatter_list}

        self.assertDictEqual(z_gather, z_gather_th)

    def test_04_clean_autogather(self):
        '''
        Test autogather cleaning
        '''
        driver_name = 'coupling_scatter'
        builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                             mod_id=self.sub_proc)
        # builder_list is a list of builders from self.sub_proc
        dict_values = {}
        # User fill in the fields in the GUI

        scatter_list = self.exec_eng.factory.create_multi_instance_driver(driver_name, builder_list)

        self.exec_eng.factory.set_builders_to_coupling_builder(scatter_list)

        self.exec_eng.configure()

        samples_df = pd.DataFrame({'selected_scenario': [True, True],
                                   'scenario_name': ['scatter1',
                                                     'scatter2']})
        dict_values[f'{self.study_name}.{driver_name}.display_options'] = {
            'group_scenarios_under_disciplines': False,
            'autogather': True}
        dict_values[f'{self.study_name}.{driver_name}.samples_df'] = samples_df
        # User fill in the fields in the GUI

        self.exec_eng.load_study_from_input_dict(dict_values)

        samples_df = pd.DataFrame({'selected_scenario': [False, True],
                                   'scenario_name': ['scatter1',
                                                     'scatter2']})

        dict_values[f'{self.study_name}.{driver_name}.samples_df'] = samples_df
        # User fill in the fields in the GUI

        self.exec_eng.load_study_from_input_dict(dict_values)

        samples_df = pd.DataFrame({'selected_scenario': [True, True],
                                   'scenario_name': ['scatter1',
                                                     'scatter2']})

        dict_values[f'{self.study_name}.{driver_name}.samples_df'] = samples_df
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
                       f'\t\t|_ scatter2',
                       f'\t\t\t|_ Disc1',
                       f'\t\t\t|_ Disc2',
                       f'\t\t|_ scatter1',
                       f'\t\t\t|_ Disc1',
                       f'\t\t\t|_ Disc2',
                       f'\t|_ {driver_name}_gather',
                       ]

        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes(exec_display=True)

        # exec treeview
        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       f'\t|_ {driver_name}',
                       f'\t\t|_ scatter2',
                       f'\t\t\t|_ Disc1',
                       f'\t\t\t|_ Disc2',
                       f'\t\t|_ scatter1',
                       f'\t\t\t|_ Disc1',
                       f'\t\t\t|_ Disc2', ]

        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()
        self.exec_eng.execute()

        y_gather = self.exec_eng.dm.get_value('MyCase.coupling_scatter_gather.y_gather')
        y_gather_th = {scatter_name: self.exec_eng.dm.get_value(
            f'{self.study_name}.{driver_name}.{scatter_name}.y') for scatter_name in scatter_list}

        self.assertDictEqual(y_gather, y_gather_th)

        indicator_gather = self.exec_eng.dm.get_value('MyCase.coupling_scatter_gather.indicator_gather')
        indicator_gather_th = {f'{scatter_name}.Disc1': self.exec_eng.dm.get_value(
            f'{self.study_name}.{driver_name}.{scatter_name}.Disc1.indicator') for scatter_name in scatter_list}

        self.assertDictEqual(indicator_gather, indicator_gather_th)

        z_gather = self.exec_eng.dm.get_value('MyCase.coupling_scatter_gather.z_gather')
        z_gather_th = {scatter_name: self.exec_eng.dm.get_value(
            f'{self.study_name}.{driver_name}.{scatter_name}.z') for scatter_name in scatter_list}

        self.assertDictEqual(z_gather, z_gather_th)
