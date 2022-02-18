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
from sos_trades_core.execution_engine.sos_multi_scatter_builder import SoSMultiScatterBuilderException


class TestMultiScatterBuilder(unittest.TestCase):
    """
    SoSMultiScatterBuilder test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''

        self.name = 'MyCase'
        self.study_name = f'{self.name}'
        self.exec_eng = ExecutionEngine(self.name)
        self.factory = self.exec_eng.factory
        self.repo = 'sos_trades_core.sos_processes.test'
        self.sub_proc = 'test_disc1_disc2_coupling'
        base_path = 'sos_trades_core.sos_wrapping.test_discs'
        self.mod1_path = f'{base_path}.disc1.Disc1'
        self.mod2_path = f'{base_path}.disc2.Disc2'

    def test_01_multibuilder_scatter(self):

        # load process in GUI
        mydict = {'input_name': 'name_list',
                  'input_type': 'string_list',
                  'input_ns': 'ns_barrierr',
                  'output_name': 'ac_name',
                  'scatter_ns': 'ns_ac',
                  'gather_ns': 'ns_barrierr'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.smaps_manager.add_build_map('name_list', mydict)
        self.exec_eng.ns_manager.add_ns('ns_barrierr', 'MyCase')

        # Create scatter map for all value blocks
        vb_dict_map = {'input_name': 'vb_dict',
                       'input_type': 'dict',
                       'input_ns': 'ns_barrierr',
                       'scatter_ns': 'ns_ac'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.smaps_manager.add_build_map('vb_dict_map', vb_dict_map)

        # instantiate factory # get instantiator from Discipline class
        cls_list = self.factory.get_builder_from_process(repo=self.repo,
                                                         mod_id=self.sub_proc)  # get instantiator from Process
        scatter_list = self.exec_eng.factory.create_value_block_builder(
            'Business', 'vb_dict_map', 'name_list', cls_list, autogather=True)

        self.exec_eng.factory.set_builders_to_coupling_builder(scatter_list)
        self.exec_eng.configure()

        # User fill in the fields in the GUI
        dict_values = {self.study_name +
                       '.vb_dict': {'actor1': {'name_1': ['Disc1', 'Disc2'], 'name_2': ['Disc1', 'Disc2']},
                                    'actor2': {'name_3': ['Disc1', 'Disc2']}}}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        constant1 = 10
        constant2 = 20
        power1 = 2
        power2 = 3

        private_val = {}
        private_val[self.study_name +
                    '.Business.actor1.Disc2.name_1.constant'] = constant1
        private_val[self.study_name +
                    '.Business.actor1.Disc2.name_1.power'] = power1
        private_val[self.study_name +
                    '.Business.actor1.Disc2.name_2.constant'] = constant2
        private_val[self.study_name +
                    '.Business.actor1.Disc2.name_2.power'] = power2

        x1 = 2
        a1 = 3
        b1 = 4
        x2 = 4
        a2 = 6
        b2 = 2
        private_val[self.study_name + '.name_1.x'] = x1
        private_val[self.study_name + '.name_2.x'] = x2
        private_val[self.study_name + '.Business.actor1.Disc1.name_1.a'] = a1
        private_val[self.study_name + '.Business.actor1.Disc1.name_2.a'] = a2
        private_val[self.study_name + '.Business.actor1.Disc1.name_1.b'] = b1
        private_val[self.study_name + '.Business.actor1.Disc1.name_2.b'] = b2
        private_val[self.study_name +
                    '.Business.actor2.Disc2.name_3.constant'] = constant2
        private_val[self.study_name +
                    '.Business.actor2.Disc2.name_3.power'] = power2

        x1 = 2
        a1 = 3
        b1 = 4
        x2 = 4
        a2 = 6
        b2 = 2
        private_val[self.study_name + '.name_3.x'] = x1
        private_val[self.study_name + '.Business.actor2.Disc1.name_3.a'] = a1
        private_val[self.study_name + '.Business.actor2.Disc1.name_3.b'] = b1

        self.exec_eng.load_study_from_input_dict(private_val)

        self.exec_eng.execute()

        y1 = self.exec_eng.dm.get_value(self.study_name + '.name_1.y')
        y2 = self.exec_eng.dm.get_value(self.study_name + '.name_2.y')
        y3 = self.exec_eng.dm.get_value(self.study_name + '.name_3.y')
        self.assertEqual(y1, a1 * x1 + b1)
        self.assertEqual(y3, a1 * x1 + b1)

        z1 = self.exec_eng.dm.get_value(self.study_name + '.name_1.z')
        z2 = self.exec_eng.dm.get_value(self.study_name + '.name_2.z')
        z3 = self.exec_eng.dm.get_value(self.study_name + '.name_3.z')
        self.assertEqual(z1, constant1 + y1**power1)
        self.assertEqual(z3, constant2 + y3**power2)

        z_dict = self.exec_eng.dm.get_value(
            self.study_name + '.Business.actor1.z_dict')
        # Check gather disciplines
        self.assertDictEqual(z_dict, {'name_1': z1, 'name_2': z2})

        y_dict = self.exec_eng.dm.get_value(
            self.study_name + '.Business.actor1.y_dict')
        # Check gather disciplines
        self.assertDictEqual(y_dict, {'name_1': y1, 'name_2': y2})

        z_dict = self.exec_eng.dm.get_value(
            self.study_name + '.Business.actor2.z_dict')
        # Check gather disciplines
        self.assertDictEqual(z_dict, {'name_3': z3})

        y_dict = self.exec_eng.dm.get_value(
            self.study_name + '.Business.actor2.y_dict')
        # Check gather disciplines
        self.assertDictEqual(y_dict, {'name_3': y3})

        self.assertFalse(self.exec_eng.dm.get_data(
            self.study_name + '.name_list', 'editable'))
        self.assertListEqual(self.exec_eng.dm.get_value(
            self.study_name + '.Business.actor1.name_list_actor1'), ['name_1', 'name_2'])
        self.assertListEqual(self.exec_eng.dm.get_value(
            self.study_name + '.Business.actor2.name_list_actor2'), ['name_3'])

    def test_02_multi_scenarios_of_multibuilder_scatter(self):

        # load process in GUI
        scenario_map = {'input_name': 'scenario_list',
                        'input_type': 'string_list',
                        'input_ns': 'ns_scatter_scenario',
                        'output_name': 'scenario_name',
                        'scatter_ns': 'ns_scenario',
                        'gather_ns': 'ns_ac',
                        'ns_to_update': ['ns_barrierr']}

        self.exec_eng.smaps_manager.add_build_map(
            'scenario_list', scenario_map)
        self.exec_eng.ns_manager.add_ns(
            'ns_scatter_scenario', 'MyCase.multi_scenarios')

        mydict = {'input_name': 'name_list',
                  'input_type': 'string_list',
                  'input_ns': 'ns_barrierr',
                  'output_name': 'ac_name',
                  'scatter_ns': 'ns_ac',
                  'gather_ns': 'ns_barrierr'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.smaps_manager.add_build_map('name_list', mydict)
        self.exec_eng.ns_manager.add_ns('ns_barrierr', 'MyCase')
        self.exec_eng.ns_manager.add_ns('ns_public', 'MyCase')

        # Create scatter map for all value blocks
        vb_dict_map = {'input_name': 'vb_dict',
                       'input_type': 'dict',
                       'input_ns': 'ns_barrierr',
                       'scatter_ns': 'ns_ac'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.smaps_manager.add_build_map('vb_dict_map', vb_dict_map)

        cls_list = self.factory.get_builder_from_process(repo=self.repo,
                                                         mod_id=self.sub_proc)  # get instantiator from Process
        scatter_list = self.exec_eng.factory.create_value_block_builder(
            'Business', 'vb_dict_map', 'name_list', cls_list, autogather=True, builder_child_path=None)

        multi_scenarios = self.exec_eng.factory.create_multi_scenario_builder(
            'multi_scenarios', 'scenario_list', [scatter_list], autogather=True, gather_node='Post-processing', business_post_proc=True)

        self.exec_eng.factory.set_builders_to_coupling_builder(multi_scenarios)
        self.exec_eng.configure()

        # User fill in the fields in the GUI
        self.study_name = 'MyCase.multi_scenarios'
        trade_variables_dict = {'vb_dict': 'dict'}

        dict_values = {f'{self.study_name}.trade_variables': trade_variables_dict,
                       f'{self.study_name}.vb_dict_trade': [{'actor1': {'name_1': ['Disc1', 'Disc2'], 'name_2': ['Disc1', 'Disc2']}, 'actor2': {'name_3': ['Disc1', 'Disc2']}},
                                                            {'actor1': {'name_1': ['Disc1', 'Disc2']}}]}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        constant1 = 10
        constant2 = 20
        power1 = 2
        power2 = 3
        x1 = 2
        a1 = 3
        b1 = 4
        x2 = 4
        a2 = 6
        b2 = 2

        private_val = {}

        for scenario in ['scenario_1', 'scenario_2']:

            private_val[self.study_name +
                        f'.{scenario}.Business.actor1.Disc2.name_1.constant'] = constant1
            private_val[self.study_name +
                        f'.{scenario}.Business.actor1.Disc2.name_1.power'] = power1
            private_val[self.study_name +
                        f'.{scenario}.Business.actor1.Disc2.name_2.constant'] = constant2
            private_val[self.study_name +
                        f'.{scenario}.Business.actor1.Disc2.name_2.power'] = power2

            private_val[self.study_name + f'.{scenario}.name_1.x'] = x1
            private_val[self.study_name + f'.{scenario}.name_2.x'] = x2
            private_val[self.study_name +
                        f'.{scenario}.Business.actor1.Disc1.name_1.a'] = a1
            private_val[self.study_name +
                        f'.{scenario}.Business.actor1.Disc1.name_2.a'] = a2
            private_val[self.study_name +
                        f'.{scenario}.Business.actor1.Disc1.name_1.b'] = b1
            private_val[self.study_name +
                        f'.{scenario}.Business.actor1.Disc1.name_2.b'] = b2
            # self.exec_eng.dm.set_values_from_dict(private_val)

            private_val[self.study_name +
                        f'.{scenario}.name_list'] = ['name_1', 'name_3']

            private_val[self.study_name +
                        f'.{scenario}.Business.actor2.Disc2.name_3.constant'] = constant2
            private_val[self.study_name +
                        f'.{scenario}.Business.actor2.Disc2.name_3.power'] = power2

            private_val[self.study_name + f'.{scenario}.name_3.x'] = x1
            private_val[self.study_name +
                        f'.{scenario}.Business.actor2.Disc1.name_3.a'] = a1
            private_val[self.study_name +
                        f'.{scenario}.Business.actor2.Disc1.name_3.b'] = b1

        self.exec_eng.load_study_from_input_dict(private_val)
        self.exec_eng.display_treeview_nodes()

        self.exec_eng.execute()

        self.exec_eng.display_treeview_nodes()

        scenario_dict = self.exec_eng.dm.get_value(
            'MyCase.Post-processing.Business.scenario_dict')

        self.assertDictEqual(scenario_dict, {'scenario_1': {'vb_dict': {'actor1': {'name_1': ['Disc1', 'Disc2'], 'name_2': ['Disc1', 'Disc2'
                                                                                                                            ]}, 'actor2': {'name_3': ['Disc1', 'Disc2']}}}, 'scenario_2': {'vb_dict': {'actor1': {'name_1': ['Disc1', 'Disc2']}}}})

    def test_03_modify_vb_dict_multibuilder_scatter(self):

        # load process in GUI
        mydict = {'input_name': 'name_list',
                  'input_type': 'string_list',
                  'input_ns': 'ns_barrierr',
                  'output_name': 'ac_name',
                  'scatter_ns': 'ns_ac',
                  'gather_ns': 'ns_barrierr'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.smaps_manager.add_build_map('name_list', mydict)
        self.exec_eng.ns_manager.add_ns('ns_barrierr', 'MyCase')

        # Create scatter map for all value blocks
        vb_dict_map = {'input_name': 'vb_dict',
                       'input_type': 'dict',
                       'input_ns': 'ns_barrierr',
                       'scatter_ns': 'ns_ac'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.smaps_manager.add_build_map('vb_dict_map', vb_dict_map)

        # instantiate factory # get instantiator from Discipline class
        cls_list = self.factory.get_builder_from_process(repo=self.repo,
                                                         mod_id=self.sub_proc)  # get instantiator from Process
        scatter_list = self.exec_eng.factory.create_value_block_builder(
            'Business', 'vb_dict_map', 'name_list', cls_list, autogather=True)

        self.exec_eng.factory.set_builders_to_coupling_builder(scatter_list)
        self.exec_eng.configure()

        # User fill in the fields in the GUI
        dict_values = {self.study_name +
                       '.vb_dict': {'actor1': {'name_1': ['Disc1', 'Disc2'], 'name_2': ['Disc1', 'Disc2']},
                                    'actor2': {'name_3': ['Disc1', 'Disc2']}}}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        constant1 = 10
        constant2 = 20
        power1 = 2
        power2 = 3

        private_val = {}
        private_val[self.study_name +
                    '.Business.actor1.Disc2.name_1.constant'] = constant1
        private_val[self.study_name +
                    '.Business.actor1.Disc2.name_1.power'] = power1
        private_val[self.study_name +
                    '.Business.actor1.Disc2.name_2.constant'] = constant2
        private_val[self.study_name +
                    '.Business.actor1.Disc2.name_2.power'] = power2

        x1 = 2
        a1 = 3
        b1 = 4
        x2 = 4
        a2 = 6
        b2 = 2
        private_val[self.study_name + '.name_1.x'] = x1
        private_val[self.study_name + '.name_2.x'] = x2
        private_val[self.study_name + '.Business.actor1.Disc1.name_1.a'] = a1
        private_val[self.study_name + '.Business.actor1.Disc1.name_2.a'] = a2
        private_val[self.study_name + '.Business.actor1.Disc1.name_1.b'] = b1
        private_val[self.study_name + '.Business.actor1.Disc1.name_2.b'] = b2
        private_val[self.study_name +
                    '.name_list'] = ['name_1', 'name_3']

        private_val[self.study_name +
                    '.Business.actor2.Disc2.name_3.constant'] = constant2
        private_val[self.study_name +
                    '.Business.actor2.Disc2.name_3.power'] = power2

        x1 = 2
        a1 = 3
        b1 = 4
        x2 = 4
        a2 = 6
        b2 = 2
        private_val[self.study_name + '.name_3.x'] = x1
        private_val[self.study_name + '.Business.actor2.Disc1.name_3.a'] = a1
        private_val[self.study_name + '.Business.actor2.Disc1.name_3.b'] = b1

        len_old_discipline_dict = len(self.exec_eng.dm.disciplines_dict)

        self.exec_eng.load_study_from_input_dict(private_val)
        self.exec_eng.display_treeview_nodes()
        self.exec_eng.execute()

        dict_values = {self.study_name +
                       '.vb_dict': {'actor1': {'name_1': ['Disc1', 'Disc2']},
                                    'actor2': {'name_3': ['Disc1', 'Disc2']}}}

        self.exec_eng.load_study_from_input_dict(dict_values)

        self.exec_eng.display_treeview_nodes()
        new_discipline_dict = self.exec_eng.dm.disciplines_dict

        # 2 disciplines have been erased (Disc1.name2 and Disc2.name2)
        self.assertEqual(len_old_discipline_dict - 2, len(new_discipline_dict))

        dict_values = {self.study_name +
                       '.vb_dict': {'actor1': {'name_1': ['Disc1', 'Disc2'], 'name_2': ['Disc1', 'Disc2'], 'name_5': ['Disc1', 'Disc2']},
                                    'actor2': {'name_3': ['Disc1', 'Disc2']}}}

        self.exec_eng.load_study_from_input_dict(dict_values)

        self.exec_eng.display_treeview_nodes()
        new_discipline_dict = self.exec_eng.dm.disciplines_dict

        # 2 disciplines have been added (Disc1.name5 and Disc2.name5)
        self.assertEqual(len_old_discipline_dict + 2, len(new_discipline_dict))

        dict_values = {self.study_name +
                       '.vb_dict': {'actor1': {'name_1': ['Disc1', 'Disc2'], 'name_2': ['Disc1', 'Disc2']},
                                    'actor2': {'name_3': ['Disc1', 'Disc2']},
                                    'actor3': {'name_6': ['Disc1', 'Disc2']}}}

        self.exec_eng.load_study_from_input_dict(dict_values)

        self.exec_eng.display_treeview_nodes()
        new_discipline_dict = self.exec_eng.dm.disciplines_dict

        # 6 disciplines have been added (two scatter +two gather+ two disciplines under
        # the two scatters)
        self.assertEqual(len_old_discipline_dict + 6, len(new_discipline_dict))

        dict_values = {self.study_name +
                       '.vb_dict': {'actor2': {'name_3': ['Disc1', 'Disc2']}}}

        self.exec_eng.load_study_from_input_dict(dict_values)

        self.exec_eng.display_treeview_nodes()
        new_discipline_dict = self.exec_eng.dm.disciplines_dict

        # 8 disciplines have been erased (2 scatters + 2 gathers and 4 disciplines under
        # the two scatters)
        self.assertEqual(len_old_discipline_dict - 8, len(new_discipline_dict))

    def test_04_clean_non_activated_value_blocks(self):

        # load process in GUI
        scenario_map = {'input_name': 'scenario_list',
                        'input_type': 'string_list',
                        'input_ns': 'ns_scatter_scenario',
                        'output_name': 'scenario_name',
                        'scatter_ns': 'ns_scenario',
                        'gather_ns': 'ns_scatter_scenario',
                        'ns_to_update': ['ns_barrierr']}

        self.exec_eng.smaps_manager.add_build_map(
            'scenario_list', scenario_map)
        self.exec_eng.ns_manager.add_ns(
            'ns_scatter_scenario', 'MyCase.multi_scenarios')

        mydict = {'input_name': 'name_list',
                  'input_type': 'string_list',
                  'input_ns': 'ns_barrierr',
                  'output_name': 'ac_name',
                  'scatter_ns': 'ns_ac',
                  'gather_ns': 'ns_barrierr'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.smaps_manager.add_build_map('name_list', mydict)
        self.exec_eng.ns_manager.add_ns('ns_barrierr', 'MyCase')
        self.exec_eng.ns_manager.add_ns('ns_public', 'MyCase')

        # Create scatter map for all value blocks
        vb_dict_map = {'input_name': 'vb_dict',
                       'input_type': 'dict',
                       'input_ns': 'ns_barrierr',
                       'scatter_ns': 'ns_ac'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.smaps_manager.add_build_map('vb_dict_map', vb_dict_map)

        cls_list = self.factory.get_builder_from_process(repo=self.repo,
                                                         mod_id=self.sub_proc)  # get instantiator from Process
        scatter_list = self.exec_eng.factory.create_value_block_builder(
            'Business', 'vb_dict_map', 'name_list', cls_list, autogather=True, builder_child_path=None)

        multi_scenarios = self.exec_eng.factory.create_multi_scenario_builder(
            'multi_scenarios', 'scenario_list', [scatter_list], autogather=True, gather_node='Post-processing', business_post_proc=True)

        self.exec_eng.factory.set_builders_to_coupling_builder(multi_scenarios)
        self.exec_eng.configure()

        # User fill in the fields in the GUI
        self.study_name = 'MyCase.multi_scenarios'
        trade_variables_dict = {'vb_dict': 'dict'}

        dict_values = {f'{self.study_name}.trade_variables': trade_variables_dict,
                       f'{self.study_name}.vb_dict_trade': [{'actor1': {'name_1': ['Disc1']}},
                                                            {'actor1': {'name_1': ['Disc1', 'Disc2'], 'name_2': ['Disc1']}, 'actor2': {'name_3': ['Disc2']}}]}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        constant1 = 10
        constant2 = 20
        power1 = 2
        power2 = 3
        x1 = 2
        a1 = 3
        b1 = 4
        x2 = 4
        a2 = 6
        b2 = 2

        private_val = {}

        for scenario in ['scenario_1', 'scenario_2']:

            private_val[self.study_name +
                        f'.{scenario}.Business.actor1.Disc2.name_1.constant'] = constant1
            private_val[self.study_name +
                        f'.{scenario}.Business.actor1.Disc2.name_1.power'] = power1
            private_val[self.study_name +
                        f'.{scenario}.Business.actor1.Disc2.name_2.constant'] = constant2
            private_val[self.study_name +
                        f'.{scenario}.Business.actor1.Disc2.name_2.power'] = power2

            private_val[self.study_name + f'.{scenario}.name_1.x'] = x1
            private_val[self.study_name + f'.{scenario}.name_2.x'] = x2
            private_val[self.study_name +
                        f'.{scenario}.Business.actor1.Disc1.name_1.a'] = a1
            private_val[self.study_name +
                        f'.{scenario}.Business.actor1.Disc1.name_2.a'] = a2
            private_val[self.study_name +
                        f'.{scenario}.Business.actor1.Disc1.name_1.b'] = b1
            private_val[self.study_name +
                        f'.{scenario}.Business.actor1.Disc1.name_2.b'] = b2
            private_val[self.study_name +
                        f'.{scenario}.name_list'] = ['name_1', 'name_3']

            private_val[self.study_name +
                        f'.{scenario}.Business.actor2.Disc2.name_3.constant'] = constant2
            private_val[self.study_name +
                        f'.{scenario}.Business.actor2.Disc2.name_3.power'] = power2
            private_val[self.study_name +
                        f'.scenario_2.name_3.y'] = 2.0

            private_val[self.study_name + f'.{scenario}.name_3.x'] = x1
            private_val[self.study_name +
                        f'.{scenario}.Business.actor2.Disc1.name_3.a'] = a1
            private_val[self.study_name +
                        f'.{scenario}.Business.actor2.Disc1.name_3.b'] = b1

        self.exec_eng.load_study_from_input_dict(private_val)
        self.exec_eng.display_treeview_nodes()

        self.exec_eng.execute()

        scenario_dict = self.exec_eng.dm.get_value(
            'MyCase.Post-processing.Business.scenario_dict')

        # test on scenario_dict value
        self.assertDictEqual(scenario_dict, {'scenario_1': {'vb_dict': {'actor1': {'name_1': ['Disc1']}}}, 'scenario_2': {'vb_dict': {'actor1': {'name_1': ['Disc1', 'Disc2'], 'name_2': [
                             'Disc1']}, 'actor2': {'name_3': ['Disc2']}}}})

        # test associated_inputs updated namespaces (name_list)
        self.assertListEqual(list(self.exec_eng.dm.get_all_namespaces_from_var_name('name_list')), list(
            ['MyCase.multi_scenarios.scenario_1.name_list', 'MyCase.multi_scenarios.scenario_2.name_list']))
        self.assertEqual(self.exec_eng.dm.get_data(
            'MyCase.multi_scenarios.scenario_1.name_list', 'namespace'), 'ns_scenario')
        self.assertEqual(self.exec_eng.dm.get_data(
            'MyCase.multi_scenarios.scenario_2.name_list', 'namespace'), 'ns_scenario')
        name_list_scen_1 = self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.name_list')
        name_list_scen_1.sort()
        self.assertListEqual(name_list_scen_1, list(
            ['name_1']))
        name_list_scen_2 = self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.name_list')
        name_list_scen_2.sort()
        self.assertListEqual(name_list_scen_2, list(
            ['name_1', 'name_2', 'name_3']))

        # vb_dict_trade modification in DM
        dict_values[f'{self.study_name}.vb_dict_trade'] = [{'actor1': {'name_1': ['Disc1', 'Disc2'], 'name_2': []}, 'actor2': {'name_3': ['Disc1']}},
                                                           {'actor1': {'name_1': ['Disc2']}}]

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        scenario_dict = self.exec_eng.dm.get_value(
            'MyCase.Post-processing.Business.scenario_dict')
        self.assertDictEqual(scenario_dict, {'scenario_1': {'vb_dict': {'actor1': {'name_1': ['Disc1', 'Disc2'], 'name_2': []}, 'actor2': {'name_3': ['Disc1']}}}, 'scenario_2': {'vb_dict': {'actor1': {'name_1': ['Disc2']}}}}
                             )

        vb_dict_scen_1 = self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.vb_dict')
        self.assertDictEqual(vb_dict_scen_1, {'actor1': {'name_1': [
                             'Disc1', 'Disc2'], 'name_2': []}, 'actor2': {'name_3': ['Disc1']}})

        vb_dict_scen_2 = self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.vb_dict')
        self.assertDictEqual(vb_dict_scen_2, {'actor1': {'name_1': ['Disc2']}})

        # vb_dict_trade modification in DM
        dict_values[f'{self.study_name}.vb_dict_trade'] = [{'actor1': {'name_1': ['Disc1', 'Disc2']}, 'actor2': {'name_2': ['Disc1']}},
                                                           {'actor3': {'name_2': ['Disc1']}}]

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        scenario_dict = self.exec_eng.dm.get_value(
            'MyCase.Post-processing.Business.scenario_dict')

        # vb_dict_trade modification in DM
        dict_values[f'{self.study_name}.vb_dict_trade'] = [{'actor3': {
            'name_1': ['Disc1', 'Disc2'], 'name_2': ['Disc2']}}]

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        scenario_dict = self.exec_eng.dm.get_value(
            'MyCase.Post-processing.Business.scenario_dict')
        self.assertDictEqual(scenario_dict, {'scenario_1': {'vb_dict': {'actor3': {'name_1': [
                             'Disc1', 'Disc2'], 'name_2': ['Disc2']}}}})

        # test on cleaning after re-configure
        self.assertNotIn(
            'MyCase.multi_scenarios.scenario_2.vb_dict', self.exec_eng.dm.data_id_map)

    def test_05_modify_vb_dict_multibuilder_scatter_with_error(self):

        # load process in GUI
        mydict = {'input_name': 'name_list',
                  'input_type': 'string_list',
                  'input_ns': 'ns_barrierr',
                  'output_name': 'ac_name',
                  'scatter_ns': 'ns_ac',
                  'gather_ns': 'ns_barrierr'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.smaps_manager.add_build_map('name_list', mydict)
        self.exec_eng.ns_manager.add_ns('ns_barrierr', 'MyCase')

        # Create scatter map for all value blocks
        vb_dict_map = {'input_name': 'vb_dict',
                       'input_type': 'dict',
                       'input_ns': 'ns_barrierr',
                       'scatter_ns': 'ns_ac'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.smaps_manager.add_build_map('vb_dict_map', vb_dict_map)

        # instantiate factory # get instantiator from Discipline class
        cls_list = self.factory.get_builder_from_process(repo=self.repo,
                                                         mod_id=self.sub_proc)  # get instantiator from Process
        scatter_list = self.exec_eng.factory.create_value_block_builder(
            'Business', 'vb_dict_map', 'name_list', cls_list, autogather=True)

        self.exec_eng.factory.set_builders_to_coupling_builder(scatter_list)
        self.exec_eng.configure()

        # User fill in the fields in the GUI
        dict_values = {self.study_name +
                       '.vb_dict': {'actor1': {'name_1': ['Disc1', 'Disc2']}}}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        constant1 = 10
        constant2 = 20
        power1 = 2
        power2 = 3

        private_val = {}
        private_val[self.study_name +
                    '.Business.actor1.Disc2.name_1.constant'] = constant1
        private_val[self.study_name +
                    '.Business.actor1.Disc2.name_1.power'] = power1
        private_val[self.study_name +
                    '.Business.actor1.Disc2.name_2.constant'] = constant2
        private_val[self.study_name +
                    '.Business.actor1.Disc2.name_2.power'] = power2

        x1 = 2
        a1 = 3
        b1 = 4
        x2 = 4
        a2 = 6
        b2 = 2
        private_val[self.study_name + '.name_1.x'] = x1
        private_val[self.study_name + '.name_2.x'] = x2
        private_val[self.study_name + '.Business.actor1.Disc1.name_1.a'] = a1
        private_val[self.study_name + '.Business.actor1.Disc1.name_2.a'] = a2
        private_val[self.study_name + '.Business.actor1.Disc1.name_1.b'] = b1
        private_val[self.study_name + '.Business.actor1.Disc1.name_2.b'] = b2
        private_val[self.study_name +
                    '.name_list'] = ['name_1', 'name_3']

        private_val[self.study_name +
                    '.Business.actor2.Disc2.name_3.constant'] = constant2
        private_val[self.study_name +
                    '.Business.actor2.Disc2.name_3.power'] = power2

        x1 = 2
        a1 = 3
        b1 = 4
        x2 = 4
        a2 = 6
        b2 = 2
        private_val[self.study_name + '.name_3.x'] = x1
        private_val[self.study_name + '.Business.actor2.Disc1.name_3.a'] = a1
        private_val[self.study_name + '.Business.actor2.Disc1.name_3.b'] = b1

        self.exec_eng.load_study_from_input_dict(private_val)
        self.exec_eng.display_treeview_nodes()
        self.exec_eng.execute()

        wrong_dict_values = {self.study_name +
                             '.vb_dict': {'actor1': 'aaa'}}

        self.assertRaises(
            SoSMultiScatterBuilderException, self.exec_eng.load_study_from_input_dict, wrong_dict_values)

        correct_dict_values = {self.study_name +
                               '.vb_dict': {'actor1': {'name_1': ['Disc1', 'Disc2'], 'name_2': ['Disc1', 'Disc2']}}}

        self.exec_eng.load_study_from_input_dict(correct_dict_values)

    def test_06_multibuilder_scatter_of_scatter(self):

        # load process in GUI
        mydict = {'input_name': 'name_list',
                  'input_type': 'string_list',
                  'input_ns': 'ns_barrierr',
                  'output_name': 'ac_name',
                  'scatter_ns': 'ns_name',
                  'gather_ns': 'ns_barrierr',
                  'ns_to_update': ['ns_ac', 'gather_sub_name']}  # or object ScatterMapBuild
        self.exec_eng.smaps_manager.add_build_map('name_list', mydict)
        self.exec_eng.ns_manager.add_ns('ns_barrierr', 'MyCase')

        # load process in GUI
        mydict = {'input_name': 'sub_name_list',
                  'input_type': 'string_list',
                  'input_ns': 'ns_actor',
                  'output_name': 'sub_name',
                  'scatter_ns': 'ns_sub_name',
                  'gather_ns': 'gather_sub_name',
                  'ns_to_update': ['ns_ac']}
        self.exec_eng.smaps_manager.add_build_map('sub_name_list', mydict)
        self.exec_eng.ns_manager.add_ns('gather_sub_name', 'MyCase.Business')
        self.exec_eng.ns_manager.add_ns('ns_out', 'MyCase')
        self.exec_eng.ns_manager.add_ns('ns_actor', 'MyCase.Business')

        # get instantiator from local Process
        builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                             mod_id=self.sub_proc)
        scatter_sub_name_list = self.exec_eng.factory.create_multi_scatter_builder_from_list(
            'sub_name_list', builder_list=builder_list, autogather=True)

        # Create scatter map for all value blocks
        vb_dict_map = {'input_name': 'vb_dict',
                       'input_type': 'dict',
                       'input_ns': 'ns_barrierr',
                       'scatter_ns': 'ns_name',
                       'ns_to_update': ['ns_ac', 'gather_sub_name'],
                       'ns_to_update_with_actor': ['ns_actor', 'gather_sub_name']}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.smaps_manager.add_build_map('vb_dict_map', vb_dict_map)

        # instantiate factory # get instantiator from Discipline class
        scatter_list = self.exec_eng.factory.create_value_block_builder(
            'Business', 'vb_dict_map', 'name_list', scatter_sub_name_list, autogather=True)

        self.exec_eng.factory.set_builders_to_coupling_builder(scatter_list)
        self.exec_eng.configure()

        # User fill in the fields in the GUI
        dict_values = {self.study_name +
                       '.vb_dict': {'actor1': {'name_1': ['Disc1', 'Disc2'], 'name_2': ['Disc1', 'Disc2']},
                                    'actor2': {'name_3': ['Disc1', 'Disc2']}},
                       self.study_name +
                       '.Business.actor1.sub_name_list': ['sub_name_1', 'sub_name_2'],
                       self.study_name +
                       '.Business.actor2.sub_name_list': ['sub_name_3']}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        constant1 = 10
        constant2 = 20
        constant3 = 30
        power1 = 2
        power2 = 3
        power3 = 3
        x1 = 2
        a1 = 3
        b1 = 4
        x2 = 4
        a2 = 6
        b2 = 2
        x3 = 5
        a3 = 6
        b3 = 6

        private_val = {}

        private_val[self.study_name +
                    '.Business.actor1.Disc2.name_1.sub_name_1.constant'] = constant1
        private_val[self.study_name +
                    '.Business.actor1.Disc2.name_1.sub_name_1.power'] = power1
        private_val[self.study_name +
                    '.Business.actor1.Disc2.name_1.sub_name_2.constant'] = constant1
        private_val[self.study_name +
                    '.Business.actor1.Disc2.name_1.sub_name_2.power'] = power1
        private_val[self.study_name +
                    '.Business.actor1.Disc2.name_2.sub_name_1.constant'] = constant2
        private_val[self.study_name +
                    '.Business.actor1.Disc2.name_2.sub_name_1.power'] = power2
        private_val[self.study_name +
                    '.Business.actor1.Disc2.name_2.sub_name_2.constant'] = constant2
        private_val[self.study_name +
                    '.Business.actor1.Disc2.name_2.sub_name_2.power'] = power2
        private_val[self.study_name +
                    '.Business.actor2.Disc2.name_3.sub_name_3.constant'] = constant3
        private_val[self.study_name +
                    '.Business.actor2.Disc2.name_3.sub_name_3.power'] = power3

        private_val[self.study_name + '.name_1.sub_name_1.x'] = x1
        private_val[self.study_name + '.name_1.sub_name_2.x'] = x1
        private_val[self.study_name + '.name_2.sub_name_1.x'] = x2
        private_val[self.study_name + '.name_2.sub_name_2.x'] = x2
        private_val[self.study_name + '.name_3.sub_name_3.x'] = x3
        private_val[self.study_name +
                    '.Business.actor1.Disc1.name_1.sub_name_1.a'] = a1
        private_val[self.study_name +
                    '.Business.actor1.Disc1.name_2.sub_name_1.a'] = a2
        private_val[self.study_name +
                    '.Business.actor1.Disc1.name_1.sub_name_2.a'] = a1
        private_val[self.study_name +
                    '.Business.actor1.Disc1.name_2.sub_name_2.a'] = a2
        private_val[self.study_name +
                    '.Business.actor1.Disc1.name_1.sub_name_1.b'] = b1
        private_val[self.study_name +
                    '.Business.actor1.Disc1.name_2.sub_name_1.b'] = b2
        private_val[self.study_name +
                    '.Business.actor1.Disc1.name_1.sub_name_2.b'] = b1
        private_val[self.study_name +
                    '.Business.actor1.Disc1.name_2.sub_name_2.b'] = b2
        private_val[self.study_name +
                    '.Business.actor2.Disc1.name_3.sub_name_3.a'] = a3
        private_val[self.study_name +
                    '.Business.actor2.Disc1.name_3.sub_name_3.b'] = b3

        self.exec_eng.load_study_from_input_dict(private_val)

        actor1_disc1_disc = self.exec_eng.dm.disciplines_id_map['MyCase.Business.actor1.Disc1']
        self.assertEqual(len(actor1_disc1_disc), 3)

        # scatter of scatter
        scatter_sub_names = self.exec_eng.dm.get_discipline(
            actor1_disc1_disc[0])
        self.assertEqual(scatter_sub_names.__class__.__name__,
                         'SoSDisciplineScatter')
        self.assertEqual(scatter_sub_names.scatter_builders.cls.__name__,
                         'SoSDisciplineScatter')

        # gather of scatter (of scatter)
        gather_sub_names = self.exec_eng.dm.get_discipline(
            actor1_disc1_disc[1])
        self.assertEqual(gather_sub_names.__class__.__name__,
                         'SoSDisciplineGather')
        self.assertEqual(gather_sub_names.builder.cls.__name__,
                         'SoSDisciplineScatter')

        # scatter of gather
        scatter_gather_sub_names = self.exec_eng.dm.get_discipline(
            actor1_disc1_disc[2])
        self.assertEqual(scatter_gather_sub_names.__class__.__name__,
                         'SoSDisciplineScatter')
        self.assertEqual(scatter_gather_sub_names.scatter_builders.cls.__name__,
                         'SoSDisciplineGather')


if '__main__' == __name__:
    cls = TestMultiScatterBuilder()
    cls.setUp()
    cls.test_02_multi_scenarios_of_multibuilder_scatter()
