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
from sos_trades_core.execution_engine.scattermaps_manager import ScatterMapsManagerException
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline


class TestGatherData(unittest.TestCase):
    """
    Gather data discipline test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.name = 'Coupling'
        self.namespace = 'MyCase'
        self.study_name = f'{self.namespace}'
        self.exec_eng = ExecutionEngine(self.namespace)
        self.factory = self.exec_eng.factory
        base_path = 'sostrades_core.sos_wrapping.test_discs'
        self.mod1_path = f'{base_path}.disc1.Disc1'
        self.mod2_path = f'{base_path}.disc2_dict.Disc2'

    def test_01_gather_data(self):

        ns_dict = {'ns_ac': self.namespace,
                   'ns_barrier': self.namespace}
        self.exec_eng.ns_manager.add_ns_def(ns_dict)

        mydict_build = {'input_name': 'name_list',
                        'input_type': 'string_list',
                        'input_ns': 'ns_barrier',
                        'output_name': 'ac_name',
                        'scatter_ns': 'ns_ac'}

        mydict_y = {'input_name': 'y',
                    'input_type': 'float',
                    'output_name': 'y_dict',
                    'output_type': 'dict',
                    'output_ns': 'ns_barrier',
                    'scatter_var_name': 'name_list'}

        mydict_x = {'input_name': 'x',
                    'input_type': 'float',
                    'output_name': 'x_dict',
                    'output_type': 'dict',
                    'output_ns': 'ns_barrier',
                    'scatter_var_name': 'name_list'}

        self.exec_eng.smaps_manager.add_build_map('name_list', mydict_build)
        self.exec_eng.smaps_manager.add_data_map('gather_y', mydict_y)
        self.exec_eng.smaps_manager.add_data_map('gather_x', mydict_x)

        disc1_builder = self.factory.get_builder_from_module(
            'Disc1', self.mod1_path)

        disc2_builder = self.factory.get_builder_from_module(
            'Disc2', self.mod2_path)

        scatter_build = self.exec_eng.factory.create_scatter_builder(
            'Disc1', 'name_list', disc1_builder)

        gather_data_y = self.exec_eng.factory.create_gather_data_builder(
            'gather_data_y', 'gather_y')

        gather_data_x = self.exec_eng.factory.create_gather_data_builder(
            'gather_data_x', 'gather_x')

        self.exec_eng.factory.set_builders_to_coupling_builder(
            [scatter_build, gather_data_y, gather_data_x, disc2_builder])
        self.exec_eng.configure()

        a = 3
        b = 4
        x1 = 2
        x2 = 4
        constant = 10
        power = 2
        # User fill in the fields in the GUI
        dict_values = {self.study_name + '.name_list': ['name_1', 'name_2'],
                       self.study_name + '.name_1.x': x1,
                       self.study_name + '.name_2.x': x2,
                       self.study_name + '.Disc1.name_1.a': a,
                       self.study_name + '.Disc1.name_2.a': a,
                       self.study_name + '.Disc1.name_1.b': b,
                       self.study_name + '.Disc1.name_2.b': b,
                       self.study_name + '.Disc2.constant': constant,
                       self.study_name + '.Disc2.power': power}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        self.exec_eng.execute()

        name_1_y = self.exec_eng.dm.get_value('MyCase.name_1.y')
        name_2_y = self.exec_eng.dm.get_value('MyCase.name_2.y')

        self.assertEqual(name_1_y, a * x1 + b)
        self.assertEqual(name_2_y, a * x2 + b)
        self.assertDictEqual(self.exec_eng.dm.get_value('MyCase.y_dict'), {
                             'name_1': name_1_y, 'name_2': name_2_y})
        self.assertDictEqual(self.exec_eng.dm.get_value('MyCase.z_dict'), {
                             'name_1': constant + name_1_y**power + x1, 'name_2': constant + name_2_y**power + x2})

        gather_discipline = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.gather_data_y')[0]
        self.assertListEqual([key for key in list(gather_discipline._data_in.keys()) if key not in SoSDiscipline.NUM_DESC_IN], [
            'name_list', 'name_1.y', 'name_2.y'])
        self.assertListEqual(
            list(gather_discipline._data_out.keys()), ['y_dict'])

        # change name_list and configure
        dict_values = {self.study_name + '.name_list': ['name_1']}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        self.exec_eng.execute()

        self.assertEqual(name_1_y, a * x1 + b)
        self.assertDictEqual(self.exec_eng.dm.get_value('MyCase.y_dict'), {
                             'name_1': name_1_y})
        self.assertDictEqual(self.exec_eng.dm.get_value('MyCase.z_dict'), {
                             'name_1': constant + name_1_y**power + x1})

        gather_discipline = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.gather_data_y')[0]
        self.assertListEqual([key for key in list(gather_discipline._data_in.keys())if key not in SoSDiscipline.NUM_DESC_IN], [
            'name_list', 'name_1.y'])
        self.assertListEqual(
            list(gather_discipline._data_out.keys()), ['y_dict'])

        # test cleaning in DM
        self.assertNotIn('MyCase.name_2.x',
                         self.exec_eng.dm.data_id_map.keys())
        self.assertNotIn('MyCase.name_2.y',
                         self.exec_eng.dm.data_id_map.keys())

        # check user_level of gather inputs
        self.assertEqual(self.exec_eng.dm.get_data(
            'MyCase.name_1.y', 'user_level'), 1)
        disc1_name_1 = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Disc1.name_1')[0]
        self.assertEqual(disc1_name_1._data_out['y']['user_level'], 1)
        gather_disc1 = self.exec_eng.dm.get_disciplines_with_name('MyCase.gather_data_y')[
            0]
        self.assertEqual(gather_disc1._data_in['name_1.y']['user_level'], 3)

    def test_02_multi_scenario_with_gather_data(self):

        ns_dict = {'ns_ac': f'{self.namespace}.multi_scenarios',
                   'ns_barrier': f'{self.namespace}.multi_scenarios'}
        self.exec_eng.ns_manager.add_ns_def(ns_dict)

        mydict_build = {'input_name': 'name_list',
                        'input_type': 'string_list',
                        'input_ns': 'ns_barrier',
                        'output_name': 'ac_name',
                        'scatter_ns': 'ns_ac'}

        mydict_y = {'input_name': 'y',
                    'input_type': 'float',
                    'output_name': 'y_dict',
                    'output_type': 'dict',
                    'output_ns': 'ns_barrier',
                    'scatter_var_name': 'name_list'}

        mydict_x = {'input_name': 'x',
                    'input_type': 'float',
                    'output_name': 'x_dict',
                    'output_type': 'dict',
                    'output_ns': 'ns_barrier',
                    'scatter_var_name': 'name_list'}

        self.exec_eng.smaps_manager.add_build_map('name_list', mydict_build)
        self.exec_eng.smaps_manager.add_data_map('gather_y', mydict_y)
        self.exec_eng.smaps_manager.add_data_map('gather_x', mydict_x)

        disc1_builder = self.factory.get_builder_from_module(
            'Disc1', self.mod1_path)

        disc2_builder = self.factory.get_builder_from_module(
            'Disc2', self.mod2_path)

        scatter_build = self.exec_eng.factory.create_scatter_builder(
            'Disc1', 'name_list', disc1_builder)

        gather_data_y = self.exec_eng.factory.create_gather_data_builder(
            'gather_data_y', 'gather_y')

        gather_data_x = self.exec_eng.factory.create_gather_data_builder(
            'gather_data_x', 'gather_x')

        builder_list = [scatter_build, gather_data_y,
                        gather_data_x, disc2_builder]

        # scenario build map
        scenario_map = {'input_name': 'scenario_list',
                        'input_type': 'string_list',
                        'input_ns': 'ns_scatter_scenario',
                        'output_name': 'scenario_name',
                        'scatter_ns': 'ns_scenario',
                        'gather_ns': 'ns_scatter_scenario',
                        'ns_to_update': ['ns_ac', 'ns_barrier']}

        self.exec_eng.smaps_manager.add_build_map(
            'scenario_list', scenario_map)
        self.exec_eng.ns_manager.add_ns(
            'ns_scatter_scenario', 'MyCase.multi_scenarios')

        multi_scenarios = self.exec_eng.factory.create_multi_scenario_builder(
            'multi_scenarios', 'scenario_list', builder_list, autogather=True)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)
        self.exec_eng.configure()

        dict_values = {f'{self.study_name}.multi_scenarios.name_list_trade': [['name_1'], ['name_1', 'name_2']],
                       f'{self.study_name}.multi_scenarios.trade_variables': {'name_list': 'string_list'}}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        a = 3
        b = 4
        x1 = 2
        x2 = 4
        constant = 10
        power = 2
        # User fill in the fields in the GUI
        dict_values = {}

        for scenario in ['scenario_1', 'scenario_2']:
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.name_1.x'] = x1
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.name_2.x'] = x2
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.Disc1.name_1.a'] = a
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.Disc1.name_2.a'] = a
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.Disc1.name_1.b'] = b
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.Disc1.name_2.b'] = b
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.Disc2.constant'] = constant
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.Disc2.power'] = power

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        self.exec_eng.execute()

        # check scenario_1 data
        name_1_y = self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.name_1.y')

        self.assertEqual(name_1_y, a * x1 + b)
        self.assertDictEqual(self.exec_eng.dm.get_value('MyCase.multi_scenarios.scenario_1.y_dict'), {
                             'name_1': name_1_y})
        self.assertDictEqual(self.exec_eng.dm.get_value('MyCase.multi_scenarios.scenario_1.z_dict'), {
                             'name_1': constant + name_1_y**power + x1})

        gather_discipline = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.multi_scenarios.scenario_1.gather_data_y')[0]
        self.assertListEqual([key for key in list(gather_discipline._data_in.keys())if key not in SoSDiscipline.NUM_DESC_IN], [
                             'name_list', 'name_1.y'])
        self.assertListEqual(
            list(gather_discipline._data_out.keys()), ['y_dict'])

        # check scenario_2 data
        name_1_y = self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.name_1.y')
        name_2_y = self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.name_2.y')

        self.assertEqual(name_1_y, a * x1 + b)
        self.assertEqual(name_2_y, a * x2 + b)
        self.assertDictEqual(self.exec_eng.dm.get_value('MyCase.multi_scenarios.scenario_2.y_dict'), {
                             'name_1': name_1_y, 'name_2': name_2_y})
        self.assertDictEqual(self.exec_eng.dm.get_value('MyCase.multi_scenarios.scenario_2.z_dict'), {
                             'name_1': constant + name_1_y**power + x1, 'name_2': constant + name_2_y**power + x2})

        gather_discipline = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.multi_scenarios.scenario_2.gather_data_y')[0]
        self.assertListEqual([key for key in list(gather_discipline._data_in.keys())if key not in SoSDiscipline.NUM_DESC_IN], [
            'name_list', 'name_1.y', 'name_2.y'])
        self.assertListEqual(
            list(gather_discipline._data_out.keys()), ['y_dict'])

    def test_03_check_lengths_in_gather_map(self):

        mydict = {'input_name': ['y', 'x'],
                  'input_type': ['float', 'float'],
                  'output_name': ['y_dict', 'x_dict'],
                  'output_type': ['dict'],
                  'output_ns': 'ns_barrier',
                  'scatter_var_name': 'name_list'}

        self.assertRaises(ScatterMapsManagerException,
                          self.exec_eng.smaps_manager.add_data_map, 'gather_map', mydict)

        mydict = {'input_name': ['y', 'x'],
                  'input_type': 'float',
                  'output_name': ['y_dict', 'x_dict'],
                  'output_type': ['dict', 'dict'],
                  'output_ns': 'ns_barrier',
                  'scatter_var_name': 'name_list'}

        self.assertRaises(ScatterMapsManagerException,
                          self.exec_eng.smaps_manager.add_data_map, 'gather_map', mydict)

    def test_04_gather_data_with_lists_in_maps(self):

        ns_dict = {'ns_ac': self.namespace,
                   'ns_barrier': self.namespace,
                   'ns_gather_input': self.namespace}
        self.exec_eng.ns_manager.add_ns_def(ns_dict)

        mydict_build = {'input_name': 'name_list',
                        'input_type': 'string_list',
                        'input_ns': 'ns_barrier',
                        'output_name': 'ac_name',
                        'scatter_ns': 'ns_ac'}

        mydict_gather_1 = {'input_name': ['y', 'x'],
                           'input_type': ['float', 'float'],
                           'output_name': ['y_dict', 'x_dict'],
                           'output_type': ['dict', 'dict'],
                           'output_ns': 'ns_barrier',
                           'scatter_var_name': 'name_list'}

        self.exec_eng.smaps_manager.add_build_map('name_list', mydict_build)
        self.exec_eng.smaps_manager.add_data_map(
            'gather_map_1', mydict_gather_1)

        disc1_builder = self.factory.get_builder_from_module(
            'Disc1', self.mod1_path)

        disc2_builder = self.factory.get_builder_from_module(
            'Disc2', self.mod2_path)

        scatter_build = self.exec_eng.factory.create_scatter_builder(
            'Disc1', 'name_list', disc1_builder)

        gather_data_1 = self.exec_eng.factory.create_gather_data_builder(
            'gather_data', 'gather_map_1')

        self.exec_eng.factory.set_builders_to_coupling_builder(
            [scatter_build, gather_data_1, disc2_builder])
        self.exec_eng.configure()

        a = 3
        b = 4
        x1 = 2
        x2 = 4
        constant = 10
        power = 2

        # User fill in the fields in the GUI
        dict_values = {self.study_name + '.name_list': ['name_1', 'name_2'],
                       self.study_name + '.name_1.x': x1,
                       self.study_name + '.name_2.x': x2,
                       self.study_name + '.Disc1.name_1.a': a,
                       self.study_name + '.Disc1.name_2.a': a,
                       self.study_name + '.Disc1.name_1.b': b,
                       self.study_name + '.Disc1.name_2.b': b,
                       self.study_name + '.Disc2.constant': constant,
                       self.study_name + '.Disc2.power': power}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        self.exec_eng.execute()

        name_1_y = self.exec_eng.dm.get_value('MyCase.name_1.y')
        name_2_y = self.exec_eng.dm.get_value('MyCase.name_2.y')

        self.assertEqual(name_1_y, a * x1 + b)
        self.assertEqual(name_2_y, a * x2 + b)
        self.assertDictEqual(self.exec_eng.dm.get_value('MyCase.y_dict'), {
                             'name_1': name_1_y, 'name_2': name_2_y})
        self.assertDictEqual(self.exec_eng.dm.get_value('MyCase.z_dict'), {
                             'name_1': constant + name_1_y**power + x1, 'name_2': constant + name_2_y**power + x2})

        gather_discipline = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.gather_data')[0]
        self.assertListEqual([key for key in list(gather_discipline._data_in.keys())if key not in SoSDiscipline.NUM_DESC_IN], [
            'name_list', 'name_1.y', 'name_1.x', 'name_2.y', 'name_2.x'])
        self.assertListEqual(
            list(gather_discipline._data_out.keys()), ['y_dict', 'x_dict'])
