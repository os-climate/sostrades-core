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
from time import sleep
from shutil import rmtree
from pathlib import Path
from os.path import join

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.sos_processes.test.test_multiscenario.usecase import Study
from sos_trades_core.execution_engine.sos_multi_scenario import SoSMultiScenario
from sos_trades_core.execution_engine.scatter_data import SoSScatterData
from sos_trades_core.execution_engine.sos_discipline_scatter import SoSDisciplineScatter
from tempfile import gettempdir
from sos_trades_core.tools.rw.load_dump_dm_data import DirectLoadDump
from sos_trades_core.study_manager.base_study_manager import BaseStudyManager


class TestMultiScenario(unittest.TestCase):
    """
    SoSMultiScenario test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.dirs_to_del = []
        self.namespace = 'MyCase'
        self.study_name = f'{self.namespace}'
        self.repo = 'sos_trades_core.sos_processes.test'
        self.base_path = 'sos_trades_core.sos_wrapping.test_discs'
        self.exec_eng = ExecutionEngine(self.namespace)
        self.factory = self.exec_eng.factory
        self.root_dir = gettempdir()

    def tearDown(self):

        for dir_to_del in self.dirs_to_del:
            sleep(0.5)
            if Path(dir_to_del).is_dir():
                rmtree(dir_to_del)
        sleep(0.5)

    def test_01_multi_scenario_of_scatter(self):

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
            'name_list', builder_list=builder_list, autogather=True)

        mod_list = f'{self.base_path}.disc3_scenario.Disc3'
        disc3_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc3', mod_list)
        scatter_list.append(disc3_builder)

        multi_scenarios = self.exec_eng.factory.create_multi_scenario_builder(
            'multi_scenarios', 'scenario_list', scatter_list, autogather=True, gather_node='Post-processing')

        self.exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()

        x1 = 2
        x2 = 4

        dict_values = {f'{self.study_name}.multi_scenarios.x_trade': [x1, x2],
                       f'{self.study_name}.multi_scenarios.trade_variables': {'x': 'float'}}

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

            private_val[self.study_name + '.name_1.a'] = a1
            private_val[self.study_name + '.name_2.a'] = a2
            private_val[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc1.name_1.b'] = b1
            private_val[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc1.name_2.b'] = b2
            private_val[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc3.constant'] = 3
            private_val[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc3.power'] = 2
        private_val[self.study_name +
                    '.multi_scenarios.scenario_1.Disc3.z'] = 1.2
        private_val[self.study_name +
                    '.multi_scenarios.scenario_2.Disc3.z'] = 1.5

        self.exec_eng.load_study_from_input_dict(private_val)

        self.exec_eng.execute()

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.x'), x1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.x'), x2)

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.name_1.y'), a1 * x1 + b1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.name_2.y'), a2 * x1 + b2)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.name_1.y'), a1 * x2 + b1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.name_2.y'), a2 * x2 + b2)

    def test_02_multi_scenario_of_scatter_name1_x_trade_variable(self):

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
            'name_list', builder_list=builder_list, autogather=True)

        mod_list = f'{self.base_path}.disc3_scenario.Disc3'
        disc3_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc3', mod_list)
        scatter_list.append(disc3_builder)

        multi_scenarios = self.exec_eng.factory.create_multi_scenario_builder(
            'multi_scenarios', 'scenario_list', scatter_list, autogather=True)

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

        self.exec_eng.display_treeview_nodes(display_variables='var_name')

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
                        scenario + '.Disc1.name_1.b'] = b1
            private_val[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc1.name_2.b'] = b2
            private_val[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc3.constant'] = 3
            private_val[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc3.power'] = 2
        private_val[self.study_name +
                    '.multi_scenarios.scenario_1.Disc3.z'] = 1.2
        private_val[self.study_name +
                    '.multi_scenarios.scenario_2.Disc3.z'] = 1.5

        self.exec_eng.load_study_from_input_dict(private_val)

        self.exec_eng.execute()

        x1 = 2
        x2 = 4

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.name_1.x'), x1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.name_1.x'), x2)

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.name_1.y'), a1 * x1 + b1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.name_2.y'), a2 * x2b + b2)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.name_1.y'), a1 * x2 + b1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.name_2.y'), a2 * x2b + b2)

    def test_03_multi_scenario_of_scatter_from_process(self):

        builder_process = self.exec_eng.factory.get_builder_from_process(
            self.repo, 'test_multiscenario')
        self.exec_eng.factory.set_builders_to_coupling_builder(
            builder_process)

        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()
        usecase = Study(execution_engine=self.exec_eng)
        usecase.study_name = self.namespace
        values_dict = usecase.setup_usecase()

        self.exec_eng.load_study_from_input_dict(values_dict[0])

        self.exec_eng.display_treeview_nodes(display_variables='var_name')
        self.exec_eng.execute()

    def test_04_consecutive_configure(self):

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
            'name_list', builder_list=builder_list, autogather=True)

        mod_path = f'{self.base_path}.disc3_scenario.Disc3'
        disc3_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc3', mod_path)
        scatter_list.append(disc3_builder)

        multi_scenarios = self.exec_eng.factory.create_multi_scenario_builder(
            'multi_scenarios', 'scenario_list', scatter_list, autogather=True, gather_node='Post-processing', business_post_proc=True)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()

        x1 = 2
        x2 = 4

        dict_values = {f'{self.study_name}.multi_scenarios.x_trade': [x1, x2],
                       f'{self.study_name}.multi_scenarios.trade_variables': {'x': 'float'},
                       f'{self.study_name}.multi_scenarios.name_list': ['name_1', 'name_2']}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        for disc in self.exec_eng.dm.get_disciplines_with_name('MyCase.multi_scenarios'):
            if isinstance(disc, SoSMultiScenario):
                self.assertListEqual(list(disc.get_scattered_disciplines().keys()), [
                                     'scenario_1', 'scenario_2'])

        dict_values = {self.study_name + '.multi_scenarios.x_trade': [2]}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        for disc in self.exec_eng.dm.get_disciplines_with_name('MyCase.multi_scenarios'):
            if isinstance(disc, SoSMultiScenario):
                self.assertListEqual(list(disc.get_scattered_disciplines().keys()), [
                                     'scenario_1'])

        dict_values = {
            self.study_name + '.multi_scenarios.name_list': ['name_1', 'name_2', 'name_3']}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        for disc in self.exec_eng.dm.get_disciplines_with_name('MyCase.multi_scenarios'):
            if isinstance(disc, SoSMultiScenario):
                self.assertListEqual(list(disc.get_scattered_disciplines().keys()), [
                                     'scenario_1'])
        for disc in self.exec_eng.dm.get_disciplines_with_name('MyCase.multi_scenarios.scenario_1.Disc1'):
            if isinstance(disc, SoSDisciplineScatter):
                self.assertListEqual(list(disc.get_scattered_disciplines().keys()), [
                                     'name_1', 'name_2', 'name_3'])

        dict_values = {self.study_name + '.multi_scenarios.x_trade': [2, 4],
                       self.study_name + '.multi_scenarios.name_list': ['name_1', 'name_2']}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        for disc in self.exec_eng.dm.get_disciplines_with_name('MyCase.multi_scenarios'):
            if isinstance(disc, SoSMultiScenario):
                self.assertListEqual(list(disc.get_scattered_disciplines().keys()), [
                                     'scenario_1', 'scenario_2'])
        for disc in self.exec_eng.dm.get_disciplines_with_name('MyCase.multi_scenarios.scenario_1.Disc1'):
            if isinstance(disc, SoSDisciplineScatter):
                self.assertListEqual(list(disc.get_scattered_disciplines().keys()), [
                                     'name_1', 'name_2'])

        private_val = {}
        scenario_list = ['scenario_1', 'scenario_2']
        for scenario in scenario_list:
            a1 = 3
            b1 = 4
            a2 = 6
            b2 = 2

            private_val[self.study_name + '.name_1.a'] = a1
            private_val[self.study_name + '.name_2.a'] = a2
            private_val[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc1.name_1.b'] = b1
            private_val[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc1.name_2.b'] = b2
            private_val[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc3.constant'] = 3
            private_val[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc3.power'] = 2
        private_val[self.study_name +
                    '.multi_scenarios.scenario_1.Disc3.z'] = 1.2
        private_val[self.study_name +
                    '.multi_scenarios.scenario_2.Disc3.z'] = 1.5

        self.exec_eng.load_study_from_input_dict(private_val)

        self.exec_eng.execute()

        for disc in self.exec_eng.dm.get_disciplines_with_name('MyCase.multi_scenarios'):
            if isinstance(disc, SoSMultiScenario):
                self.assertListEqual(
                    [key for key in list(disc.get_data_io_dict('in').keys()) if key not in disc.NUM_DESC_IN], ['trade_variables', 'scenario_list', 'x_trade', 'scenario_dict'])
                self.assertListEqual(self.exec_eng.dm.get_value(
                    f'{self.study_name}.multi_scenarios.x_trade'), [2, 4])
                self.assertListEqual(self.exec_eng.dm.get_value(
                    f'{self.study_name}.multi_scenarios.scenario_list'), ['scenario_1', 'scenario_2'])
                self.assertDictEqual(self.exec_eng.dm.get_value(
                    f'{self.study_name}.Post-processing.Business.scenario_dict'), {'scenario_1': {'x': 2}, 'scenario_2': {'x': 4}})

            elif isinstance(disc, SoSScatterData):
                self.assertListEqual(
                    [key for key in list(disc.get_data_io_dict('in').keys()) if key not in disc.NUM_DESC_IN], ['x_dict', 'scenario_list'])
                self.assertListEqual(
                    list(disc.get_data_io_dict('out').keys()), ['scenario_1.x', 'scenario_2.x'])
                self.assertDictEqual(self.exec_eng.dm.get_value(
                    f'{self.study_name}.multi_scenarios.x_dict'), {'scenario_1': 2, 'scenario_2': 4})
                self.assertEqual(self.exec_eng.dm.get_value(
                    f'{self.study_name}.multi_scenarios.scenario_1.x'), 2)
                self.assertEqual(self.exec_eng.dm.get_value(
                    f'{self.study_name}.multi_scenarios.scenario_2.x'), 4)

    def test_05_dump_and_load_after_execute(self):

        builders = self.exec_eng.factory.get_builder_from_process(
            repo=self.repo, mod_id='test_disc1_disc3_multi_scenario')
        self.exec_eng.factory.set_builders_to_coupling_builder(
            builders)
        self.exec_eng.configure()

        x1 = 2
        x2 = 4

        dict_values = {}

        dict_values[f'{self.study_name}.multi_scenarios.x_trade'] = [x1, x2]
        dict_values[f'{self.study_name}.multi_scenarios.trade_variables'] = {
            'x': 'float'}
        dict_values[f'{self.study_name}.multi_scenarios.name_list'] = [
            'name_1', 'name_2']

        scenario_list = ['scenario_1', 'scenario_2']
        for scenario in scenario_list:
            a1 = 3
            b1 = 4
            a2 = 6
            b2 = 2

            dict_values[self.study_name + '.name_1.a'] = a1
            dict_values[self.study_name + '.name_2.a'] = a2
            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc1.name_1.b'] = b1
            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc1.name_2.b'] = b2
            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc3.constant'] = 3
            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc3.power'] = 2
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_1.Disc3.z'] = 1.2
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_2.Disc3.z'] = 1.5

        self.exec_eng.load_study_from_input_dict(dict_values)

        self.exec_eng.execute()

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.name_1.y'), a1 * x1 + b1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.name_2.y'), a2 * x1 + b2)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.name_1.y'), a1 * x2 + b1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.name_2.y'), a2 * x2 + b2)

        dump_dir = join(self.root_dir, self.namespace)

        BaseStudyManager.static_dump_data(
            dump_dir, self.exec_eng, DirectLoadDump())

        exec_eng2 = ExecutionEngine(self.namespace)
        builders = exec_eng2.factory.get_builder_from_process(
            repo=self.repo, mod_id='test_disc1_disc3_multi_scenario')
        exec_eng2.factory.set_builders_to_coupling_builder(builders)

        exec_eng2.configure()

        BaseStudyManager.static_load_data(
            dump_dir, exec_eng2, DirectLoadDump())

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.name_1.y'), a1 * x1 + b1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.name_2.y'), a2 * x1 + b2)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.name_1.y'), a1 * x2 + b1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.name_2.y'), a2 * x2 + b2)
        # Clean the dump folder at the end of the test
        self.dirs_to_del.append(
            join(self.root_dir, self.namespace))

    def test_06_several_trade_variables(self):

        builders = self.exec_eng.factory.get_builder_from_process(
            repo=self.repo, mod_id='test_disc1_disc3_multi_scenario')
        self.exec_eng.factory.set_builders_to_coupling_builder(
            builders)
        self.exec_eng.configure()

        x1 = 2
        x2 = 4
        x3 = 0
        x4 = 3

        dict_values = {}

        dict_values[f'{self.study_name}.multi_scenarios.name_1.x_trade'] = [
            x1, x2]
        dict_values[f'{self.study_name}.multi_scenarios.name_2.x_trade'] = [
            x3, x4]
        dict_values[f'{self.study_name}.multi_scenarios.trade_variables'] = {'name_1.x': 'float',
                                                                             'name_2.x': 'float'}
        dict_values[f'{self.study_name}.multi_scenarios.name_list'] = [
            'name_1', 'name_2']

        scenario_list = ['scenario_1', 'scenario_2',
                         'scenario_3', 'scenario_4']
        for scenario in scenario_list:
            a1 = 3
            b1 = 4
            a2 = 6
            b2 = 2

            dict_values[self.study_name + '.name_1.a'] = a1
            dict_values[self.study_name + '.name_2.a'] = a2
            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc1.name_1.b'] = b1
            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc1.name_2.b'] = b2
            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc3.constant'] = 3
            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc3.power'] = 2
            dict_values[self.study_name +
                        '.multi_scenarios.' + scenario + '.Disc3.z'] = 1.2

        self.exec_eng.load_study_from_input_dict(dict_values)

        self.exec_eng.display_treeview_nodes()

        self.exec_eng.execute()

        scenario_dict = {'scenario_1': {'name_1.x': x1, 'name_2.x': x3},
                         'scenario_2': {'name_1.x': x1, 'name_2.x': x4},
                         'scenario_3': {'name_1.x': x2, 'name_2.x': x3},
                         'scenario_4': {'name_1.x': x2, 'name_2.x': x4}}

        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_dict'), scenario_dict)

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.name_1.y'), a1 * x1 + b1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.name_2.y'), a2 * x3 + b2)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.name_1.y'), a1 * x1 + b1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.name_2.y'), a2 * x4 + b2)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_3.name_1.y'), a1 * x2 + b1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_3.name_2.y'), a2 * x3 + b2)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_4.name_1.y'), a1 * x2 + b1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_4.name_2.y'), a2 * x4 + b2)

    def test_07_trade_on_name_list(self):

        builders = self.exec_eng.factory.get_builder_from_process(
            repo=self.repo, mod_id='test_disc1_disc3_multi_scenario')
        self.exec_eng.factory.set_builders_to_coupling_builder(
            builders)
        self.exec_eng.configure()

        dict_values = {}

        dict_values[f'{self.study_name}.multi_scenarios.trade_variables'] = {
            'name_list': 'list'}
        dict_values[f'{self.study_name}.multi_scenarios.name_list_trade'] = [
            ['name_1'], ['name_1', 'name_2'], ['name_1', 'name_2', 'name_3']]

        scenario_list = ['scenario_1', 'scenario_2',
                         'scenario_3']
        for scenario in scenario_list:
            a1 = 3
            b1 = 4
            a2 = 6
            b2 = 2
            a3 = 10
            b3 = 0
            x = 2

            dict_values[self.study_name + '.name_1.a'] = a1
            dict_values[self.study_name + '.name_2.a'] = a2
            dict_values[self.study_name + '.name_3.a'] = a3

            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc3.constant'] = 3
            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc3.power'] = 2
            dict_values[self.study_name +
                        '.multi_scenarios.' + scenario + '.Disc3.z'] = 1.2

        dict_values[self.study_name +
                    '.multi_scenarios.scenario_1.Disc1.name_1.b'] = b1

        dict_values[self.study_name +
                    '.multi_scenarios.scenario_2.Disc1.name_1.b'] = b1
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_2.Disc1.name_2.b'] = b2

        dict_values[self.study_name +
                    '.multi_scenarios.scenario_3.Disc1.name_1.b'] = b1
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_3.Disc1.name_2.b'] = b2
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_3.Disc1.name_3.b'] = b3

        dict_values[self.study_name + '.name_1.x'] = x
        dict_values[self.study_name + '.name_2.x'] = x
        dict_values[self.study_name + '.name_3.x'] = x

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        scenario_dict = {'scenario_1': {'name_list': ['name_1']},
                         'scenario_2': {'name_list': ['name_1', 'name_2']},
                         'scenario_3': {'name_list': ['name_1', 'name_2', 'name_3']}}

        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_dict'), scenario_dict)

        self.exec_eng.execute()

        y1 = a1 * x + b1
        y2 = a2 * x + b2
        y3 = a3 * x + b3

        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.y_dict'), {'name_1': y1})
        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.y_dict'), {'name_1': y1, 'name_2': y2})
        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_3.y_dict'), {'name_1': y1, 'name_2': y2, 'name_3': y3})
        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.y_dict'), {'scenario_1.name_1': y1,
                                               'scenario_2.name_1': y1,
                                               'scenario_2.name_2': y2,
                                               'scenario_3.name_1': y1,
                                               'scenario_3.name_2': y2,
                                               'scenario_3.name_3': y3})

    def test_08_trade_on_name_list_with_scatter_data(self):

        ee = self.exec_eng

        # scatter build map
        ac_map = {'input_name': 'name_list',

                  'input_ns': 'ns_scatter_scenario',
                  'output_name': 'ac_name',
                  'scatter_ns': 'ns_ac',
                  'gather_ns': 'ns_scenario',
                  'ns_to_update': ['ns_data_ac']}

        ee.smaps_manager.add_build_map('name_list', ac_map)

        # scenario build map
        scenario_map = {'input_name': 'scenario_list',

                        'input_ns': 'ns_scatter_scenario',
                        'output_name': 'scenario_name',
                        'scatter_ns': 'ns_scenario',
                        'gather_ns': 'ns_scatter_scenario',
                        'ns_to_update': ['ns_disc3', 'ns_barrierr', 'ns_out_disc3']}

        scatter_data_map = {'input_name': 'o_dict',
                            'input_type': 'dict',
                            'input_ns': 'ns_scenario',
                            'output_name': 'o',
                            'output_type': 'float',
                            'scatter_var_name': 'name_list'}

        ee.smaps_manager.add_build_map(
            'scenario_list', scenario_map)

        ee.smaps_manager.add_data_map(
            'scatter_data_map', scatter_data_map)

        # shared namespace
        ee.ns_manager.add_ns('ns_barrierr', 'MyCase')
        ee.ns_manager.add_ns(
            'ns_scatter_scenario', 'MyCase.multi_scenarios')
        ee.ns_manager.add_ns(
            'ns_disc3', 'MyCase.multi_scenarios.Disc3')
        ee.ns_manager.add_ns(
            'ns_out_disc3', 'MyCase.multi_scenarios')
        ee.ns_manager.add_ns(
            'ns_data_ac', 'MyCase')

        # instantiate factory # get instantiator from Discipline class

        builder_list = ee.factory.get_builder_from_process(repo='sos_trades_core.sos_processes.test',
                                                           mod_id='test_disc1_scenario')

        scatter_list = ee.factory.create_multi_scatter_builder_from_list(
            'name_list', builder_list=builder_list, autogather=True)

        scatter_data = ee.factory.create_scatter_data_builder(
            'scatter_data', 'scatter_data_map')

        mod_list = 'sos_trades_core.sos_wrapping.test_discs.disc3_dict.Disc3'
        disc3_builder = ee.factory.get_builder_from_module(
            'Disc3', mod_list)
        scatter_list.append(disc3_builder)
        scatter_list.append(scatter_data)

        multi_scenarios = ee.factory.create_multi_scenario_builder(
            'multi_scenarios', 'scenario_list', scatter_list, autogather=True, gather_node='Post-processing')

        self.exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()

        dict_values = {}

        dict_values[f'{self.study_name}.multi_scenarios.trade_variables'] = {
            'name_list': 'list'}
        dict_values[f'{self.study_name}.multi_scenarios.name_list_trade'] = [
            ['name_1'], ['name_1', 'name_2'], ['name_1', 'name_2', 'name_3']]

        scenario_list = ['scenario_1', 'scenario_2',
                         'scenario_3']
        for scenario in scenario_list:
            a1 = 3
            b1 = 4
            a2 = 6
            b2 = 2
            a3 = 10
            b3 = 0
            x = 2

            dict_values[self.study_name + '.name_1.a'] = a1
            dict_values[self.study_name + '.name_2.a'] = a2
            dict_values[self.study_name + '.name_3.a'] = a3

            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc3.constant'] = 3
            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc3.power'] = 2
            dict_values[self.study_name +
                        '.multi_scenarios.' + scenario + '.Disc3.z'] = 1.2

        dict_values[self.study_name +
                    '.multi_scenarios.scenario_1.Disc1.name_1.b'] = b1

        dict_values[self.study_name +
                    '.multi_scenarios.scenario_2.Disc1.name_1.b'] = b1
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_2.Disc1.name_2.b'] = b2

        dict_values[self.study_name +
                    '.multi_scenarios.scenario_3.Disc1.name_1.b'] = b1
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_3.Disc1.name_2.b'] = b2
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_3.Disc1.name_3.b'] = b3

        dict_values[self.study_name + '.name_1.x'] = x
        dict_values[self.study_name + '.name_2.x'] = x
        dict_values[self.study_name + '.name_3.x'] = x

        dict_values[self.study_name + '.z'] = 1.0
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        scatter_data_gather = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Post-processing.scatter_data')[0]

        keys_in = ['scenario_list', 'scenario_1.name_1.o', 'scenario_2.name_1.o',
                   'scenario_2.name_2.o', 'scenario_3.name_1.o', 'scenario_3.name_2.o', 'scenario_3.name_3.o']

        self.assertListEqual(
            [key for key in list(sorted(scatter_data_gather._data_in.keys())) if key not in scatter_data_gather.NUM_DESC_IN], sorted(keys_in))

        self.exec_eng.execute()


if '__main__' == __name__:
    cls = TestMultiScenario()
    cls.setUp()
    cls.test_08_trade_on_name_list_with_scatter_data()
