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

import pandas as pd
from sostrades_core.execution_engine.proxy_discipline_scatter import ProxyDisciplineScatter
'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import unittest
from time import sleep
from shutil import rmtree
from pathlib import Path
from os.path import join
import os

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
# from sos_trades_core.execution_engine.sos_very_simple_multi_scenario import SoSVerySimpleMultiScenario
# from sostrades_core.execution_engine.scatter_data import SoSScatterData
from tempfile import gettempdir
from sostrades_core.tools.rw.load_dump_dm_data import DirectLoadDump
from sostrades_core.study_manager.base_study_manager import BaseStudyManager
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.execution_engine.proxy_coupling import ProxyCoupling
from sostrades_core.tools.post_processing.post_processing_factory import PostProcessingFactory


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

    def test_01_multi_scenario_of_scatter(self):

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
        self.exec_eng.ns_manager.add_ns(
            'ns_post_proc', 'MyCase.Post-processing')

        # instantiate factory # get instantiator from Discipline class

        builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                             mod_id='test_disc1_scenario')

        scatter_list = self.exec_eng.factory.create_multi_scatter_builder_from_list(
            'name_list', builder_list=builder_list, autogather=False)  # TODO: handle autogather input order and set to True...

        mod_list = f'{self.base_path}.disc3_scenario.Disc3'
        disc3_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc3', mod_list)
        scatter_list.append(disc3_builder)

        multi_scenarios = self.exec_eng.factory.create_very_simple_multi_scenario_driver(
            'multi_scenarios', 'scenario_list', scatter_list, autogather=False, gather_node='Post-processing')

        # add post-processing on 'Post-processing' node by loading a module
        # with implemented graphs
        self.exec_eng.post_processing_manager.add_post_processing_module_to_namespace(
            'ns_post_proc', 'sostrades_core.sos_wrapping.test_discs.chart_post_proc_multi_scenario')

        self.exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)
        self.exec_eng.configure()

        dict_values = {}
        dict_values[f'{self.study_name}.multi_scenarios.builder_mode'] = 'multi_instance'
        dict_values[f'{self.study_name}.multi_scenarios.scenario_df'] = pd.DataFrame({'selected_scenario': [True,
                                                                                                            False,
                                                                                                            True],
                                                                                      'scenario_name': ['scenario_1',
                                                                                                        'scenario_Z',
                                                                                                        'scenario_2']})
        # dict_values[f'{self.study_name}.multi_scenarios.scenario_list'] = [
        #     'scenario_1', 'scenario_2']

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
                    '.multi_scenarios.name_list'] = ['name_1', 'name_2']
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_1.Disc3.z'] = 1.2
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_2.Disc3.z'] = 1.5
        dict_values[self.study_name + '.name_1.x'] = x1
        dict_values[self.study_name + '.name_2.x'] = x2

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        self.exec_eng.execute()

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_list'), ['scenario_1', 'scenario_2'])

        y1 = a1 * x1 + b1
        y2 = a2 * x2 + b2

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.name_1.y'), y1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.name_2.y'), y2)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.name_1.y'), y1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.name_2.y'), y2)

        # gather_disc1 = self.exec_eng.dm.get_disciplines_with_name(
        #     'MyCase.Post-processing.Disc1')[0]
        # self.assertListEqual([key for key in list(gather_disc1._data_in.keys()) if key not in gather_disc1.NUM_DESC_IN], [
        #     'scenario_list', 'scenario_1.y_dict', 'scenario_2.y_dict'])
        # self.assertListEqual(list(gather_disc1.get_data_out().keys()), ['y_dict'])
        # self.assertDictEqual(gather_disc1.get_data_out()['y_dict']['value'], {
        #                      'scenario_1.name_1': y1, 'scenario_1.name_2': y2, 'scenario_2.name_1': y1, 'scenario_2.name_2': y2})

#         gather_disc3 = self.exec_eng.dm.get_disciplines_with_name(
#             'MyCase.Post-processing.Disc3')[0]
#         self.assertListEqual([key for key in list(gather_disc3.get_data_in().keys())if key not in ProxyDiscipline.NUM_DESC_IN], [
#             'scenario_list', 'scenario_1.o', 'scenario_2.o'])
#         self.assertListEqual(
#             list(gather_disc3.get_data_out().keys()), ['o_dict'])

        # test load filters and graphs of autogather multi-scenario
        ppf = PostProcessingFactory()

#         gather_disc1 = self.exec_eng.dm.get_disciplines_with_name(
#             'MyCase.Post-processing.Disc1')[0]
#         filters = ppf.get_post_processing_filters_by_discipline(gather_disc1)
#         charts = ppf.get_post_processing_by_discipline(
#             gather_disc1, filters, as_json=False)
# #         for chart in charts:
# #             chart.to_plotly().show()

        # get post-processing by namespace
        filters_by_namespace = ppf.get_post_processing_filters_by_namespace(
            self.exec_eng, 'MyCase.Post-processing')
        charts_by_namespace = ppf.get_post_processing_by_namespace(
            self.exec_eng, 'MyCase.Post-processing', filters_by_namespace, as_json=False, for_test=True)
#         for chart in charts_by_namespace:
#             chart.to_plotly().show()

        # get all post-processing
        all_post_processings = ppf.get_all_post_processings(
            self.exec_eng, False, as_json=False, for_test=True)
#         for post_proc_list in all_post_processings.values():
#             for graph in post_proc_list[0].post_processings:
#                 graph.to_plotly().show()

    def _test_01bis_multi_scenario_of_scatter_with_tool(self):

        # scatter build map

        # scenario build map
        scenario_map = {'input_name': 'scenario_list',
                        'input_type': 'string_list',
                        'input_ns': 'ns_scatter_scenario',
                        'output_name': 'scenario_name',
                        'scatter_ns': 'ns_scenario',
                        'gather_ns': 'ns_scatter_scenario',
                        'ns_to_update': ['ns_disc3', 'ns_barrierr', 'ns_out_disc3', 'ns_ac']}

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
            'ns_ac', 'MyCase.multi_scenarios')
        self.exec_eng.ns_manager.add_ns(
            'ns_data_ac', 'MyCase')
        self.exec_eng.ns_manager.add_ns(
            'ns_post_proc', 'MyCase.Post-processing')

        # instantiate factory # get instantiator from Discipline class

        builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                             mod_id='test_disc1_scenario')

        mod_list = f'{self.base_path}.disc3_scenario.Disc3'
        disc3_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc3', mod_list)
        builder_list.append(disc3_builder)

        multi_scenarios = self.exec_eng.factory.create_scatter_driver_with_tool(
            'multi_scenarios', 'scenario_list', builder_list, autogather=False, gather_node='Post-processing')

        # add post-processing on 'Post-processing' node by loading a module
        # with implemented graphs
        self.exec_eng.post_processing_manager.add_post_processing_module_to_namespace(
            'ns_post_proc', 'sostrades_core.sos_wrapping.test_discs.chart_post_proc_multi_scenario')

        self.exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)
        self.exec_eng.configure()

        dict_values = {}
        scenario_df = pd.DataFrame({'selected_scenario': [True, True],
                                    'scenario_name': ['scenario_1',
                                                      'scenario_2']})
        dict_values[f'{self.study_name}.multi_scenarios.scenario_df'] = scenario_df

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

            dict_values[self.study_name + '.a'] = a1
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
        dict_values[self.study_name + '.x'] = x1

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        self.exec_eng.execute()

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_df')['scenario_name'].values.tolist(),  ['scenario_1',
                                                                                      'scenario_2'])

        y1 = a1 * x1 + b1
        y2 = a1 * x1 + b2

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.scenario_1.y'), y1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.scenario_2.y'), y2)

        # gather_disc1 = self.exec_eng.dm.get_disciplines_with_name(
        #     'MyCase.Post-processing.Disc1')[0]
        # self.assertListEqual([key for key in list(gather_disc1._data_in.keys()) if key not in gather_disc1.NUM_DESC_IN], [
        #     'scenario_list', 'scenario_1.y_dict', 'scenario_2.y_dict'])
        # self.assertListEqual(list(gather_disc1.get_data_out().keys()), ['y_dict'])
        # self.assertDictEqual(gather_disc1.get_data_out()['y_dict']['value'], {
        #                      'scenario_1.name_1': y1, 'scenario_1.name_2': y2, 'scenario_2.name_1': y1, 'scenario_2.name_2': y2})

#         gather_disc3 = self.exec_eng.dm.get_disciplines_with_name(
#             'MyCase.Post-processing.Disc3')[0]
#         self.assertListEqual([key for key in list(gather_disc3.get_data_in().keys())if key not in ProxyDiscipline.NUM_DESC_IN], [
#             'scenario_list', 'scenario_1.o', 'scenario_2.o'])
#         self.assertListEqual(
#             list(gather_disc3.get_data_out().keys()), ['o_dict'])

        # test load filters and graphs of autogather multi-scenario
        ppf = PostProcessingFactory()

#         gather_disc1 = self.exec_eng.dm.get_disciplines_with_name(
#             'MyCase.Post-processing.Disc1')[0]
#         filters = ppf.get_post_processing_filters_by_discipline(gather_disc1)
#         charts = ppf.get_post_processing_by_discipline(
#             gather_disc1, filters, as_json=False)
# #         for chart in charts:
# #             chart.to_plotly().show()

        # get post-processing by namespace
        filters_by_namespace = ppf.get_post_processing_filters_by_namespace(
            self.exec_eng, 'MyCase.Post-processing')
        charts_by_namespace = ppf.get_post_processing_by_namespace(
            self.exec_eng, 'MyCase.Post-processing', filters_by_namespace, as_json=False, for_test=True)
#         for chart in charts_by_namespace:
#             chart.to_plotly().show()

        # get all post-processing
        all_post_processings = ppf.get_all_post_processings(
            self.exec_eng, False, as_json=False, for_test=True)
#         for post_proc_list in all_post_processings.values():
#             for graph in post_proc_list[0].post_processings:
#                 graph.to_plotly().show()

    def test_02_consecutive_configure(self):

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
            'name_list', builder_list=builder_list, autogather=False)  # TODO: handle autogather input order and set to True...

        mod_path = f'{self.base_path}.disc3_scenario.Disc3'
        disc3_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc3', mod_path)
        scatter_list.append(disc3_builder)

        multi_scenarios = self.exec_eng.factory.create_very_simple_multi_scenario_driver(
            'multi_scenarios', 'scenario_list', scatter_list, autogather=True, gather_node='Post-processing')

        self.exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)
        self.exec_eng.configure()

        dict_values = {}
        dict_values[f'{self.study_name}.multi_scenarios.builder_mode'] = 'multi_instance'
        dict_values[f'{self.study_name}.multi_scenarios.scenario_df'] = pd.DataFrame({'selected_scenario': [True,
                                                                                                            True],
                                                                                      'scenario_name': ['scenario_1',
                                                                                                        'scenario_2']})
        dict_values[f'{self.study_name}.multi_scenarios.name_list'] = [
            'name_1', 'name_2']
        # dict_values = {f'{self.study_name}.multi_scenarios.scenario_list': ['scenario_1', 'scenario_2'],
        # f'{self.study_name}.multi_scenarios.name_list': ['name_1', 'name_2']}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        # # # check tree view with scenario_1 and scenario_2 #TODO: reactivate checks when treeview is fixed
        # exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
        #                f'|_ {self.study_name}',
        #                f'\t|_ multi_scenarios',
        #                '\t\t|_ scenario_1',
        #                '\t\t\t|_ Disc1',
        #                '\t\t\t\t|_ name_1',
        #                '\t\t\t\t|_ name_2',
        #                '\t\t\t|_ Disc3',
        #                '\t\t|_ scenario_2',
        #                '\t\t\t|_ Disc1',
        #                '\t\t\t\t|_ name_1',
        #                '\t\t\t\t|_ name_2',
        #                '\t\t\t|_ Disc3',
        #                '\t|_ Post-processing',
        #                '\t\t|_ Disc1',
        #                '\t\t|_ Disc3',
        #                '\t|_ name_1',
        #                '\t|_ name_2', ]
        # exp_tv_str = '\n'.join(exp_tv_list)
        # assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        for disc in self.exec_eng.dm.get_disciplines_with_name('MyCase.multi_scenarios'):
            if isinstance(disc, ProxyDisciplineScatter):
                self.assertListEqual(list(disc.get_scattered_disciplines().keys()), [
                                     'scenario_1', 'scenario_2'])

        dict_values[self.study_name + '.multi_scenarios.scenario_df'] = pd.DataFrame({'selected_scenario': [True,
                                                                                                            False],
                                                                                      'scenario_name': ['scenario_1',
                                                                                                        'scenario_2']})
        # dict_values[self.study_name +
        #             '.multi_scenarios.scenario_list'] = ['scenario_1']

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        # # check tree view after scenario_2 deletion to validate cleaning #TODO: reactivate checks when treeview is fixed
        # exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
        #                f'|_ {self.study_name}',
        #                f'\t|_ multi_scenarios',
        #                '\t\t|_ scenario_1',
        #                '\t\t\t|_ Disc1',
        #                '\t\t\t\t|_ name_1',
        #                '\t\t\t\t|_ name_2',
        #                '\t\t\t|_ Disc3',
        #                '\t|_ Post-processing',
        #                '\t\t|_ Disc1',
        #                '\t\t|_ Disc3',
        #                '\t|_ name_1',
        #                '\t|_ name_2', ]
        # exp_tv_str = '\n'.join(exp_tv_list)
        # assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        self.assertListEqual(
            [key for key in self.exec_eng.dm.data_id_map.keys()
             if 'scenario_2' in key and key.split('.')[-1] not in ProxyDiscipline.NUM_DESC_IN and
             key.split('.')[-1] not in ProxyCoupling.DESC_IN],
            [])

        for disc in self.exec_eng.dm.get_disciplines_with_name('MyCase.multi_scenarios'):
            if isinstance(disc, ProxyDisciplineScatter):
                self.assertListEqual(list(disc.get_scattered_disciplines().keys()), [
                                     'scenario_1'])

        dict_values[self.study_name +
                    '.multi_scenarios.scenario_df'] = pd.DataFrame({'selected_scenario': [True,
                                                                                          True,
                                                                                          True],
                                                                    'scenario_name': ['scenario_1',
                                                                                      'scenario_2',
                                                                                      'scenario_3']})
        # dict_values[self.study_name +
        #             '.multi_scenarios.scenario_list'] = ['scenario_1', 'scenario_2', 'scenario_3']

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        for disc in self.exec_eng.dm.get_disciplines_with_name('MyCase.multi_scenarios'):
            if isinstance(disc, ProxyDisciplineScatter):
                self.assertListEqual(list(disc.get_scattered_disciplines().keys()), [
                                     'scenario_1', 'scenario_2', 'scenario_3'])

        dict_values[self.study_name +
                    '.multi_scenarios.scenario_df'] = pd.DataFrame({'selected_scenario': [],
                                                                    'scenario_name': []})
        # dict_values[self.study_name +
        #             '.multi_scenarios.scenario_list'] = []

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        for disc in self.exec_eng.dm.get_disciplines_with_name('MyCase.multi_scenarios'):
            if isinstance(disc, ProxyDisciplineScatter):
                self.assertListEqual(
                    list(disc.get_scattered_disciplines().keys()), [])

        dict_values[self.study_name +
                    '.multi_scenarios.scenario_df'] = pd.DataFrame({'selected_scenario': [True,
                                                                                          True],
                                                                    'scenario_name': ['scenario_A',
                                                                                      'scenario_B']})
        # dict_values[self.study_name +
        #             '.multi_scenarios.scenario_list'] = ['scenario_A', 'scenario_B']

        self.assertListEqual(
            [key for key in self.exec_eng.dm.data_id_map.keys() if 'scenario_1' in key and key.split('.')[-1] not in ProxyDiscipline.NUM_DESC_IN and
             key.split('.')[-1] not in ProxyCoupling.DESC_IN], [])
        self.assertListEqual(
            [key for key in self.exec_eng.dm.data_id_map.keys() if 'scenario_2' in key and key.split('.')[-1] not in ProxyDiscipline.NUM_DESC_IN and
             key.split('.')[-1] not in ProxyCoupling.DESC_IN], [])
        self.assertListEqual(
            [key for key in self.exec_eng.dm.data_id_map.keys() if 'scenario_3' in key and key.split('.')[-1] not in ProxyDiscipline.NUM_DESC_IN and
             key.split('.')[-1] not in ProxyCoupling.DESC_IN], [])

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        for disc in self.exec_eng.dm.get_disciplines_with_name('MyCase.multi_scenarios'):
            if isinstance(disc, ProxyDisciplineScatter):
                self.assertListEqual(list(disc.get_scattered_disciplines().keys()), [
                                     'scenario_A', 'scenario_B'])

        scenario_list = ['scenario_A', 'scenario_B']
        for scenario in scenario_list:
            a1 = 3
            b1 = 4
            a2 = 6
            b2 = 2
            x1 = 2
            x2 = 4

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
                    '.multi_scenarios.scenario_A.Disc3.z'] = 1.2
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_B.Disc3.z'] = 1.5
        dict_values[self.study_name + '.name_1.x'] = x1
        dict_values[self.study_name + '.name_2.x'] = x2

        self.exec_eng.load_study_from_input_dict(dict_values)

        self.exec_eng.execute()

        for disc in self.exec_eng.dm.get_disciplines_with_name('MyCase.multi_scenarios'):
            if isinstance(disc, ProxyDisciplineScatter):
                # if isinstance(disc, SoSVerySimpleMultiScenario):
                self.assertListEqual(
                    [key for key in list(disc.get_data_io_dict('in').keys()) if key not in disc.NUM_DESC_IN], ['scenario_list'])
                self.assertListEqual(self.exec_eng.dm.get_value(
                    f'{self.study_name}.multi_scenarios.scenario_list'), ['scenario_A', 'scenario_B'])
                self.assertListEqual(
                    list(self.exec_eng.dm.get_disciplines_with_name(
                        f'{self.study_name}')[0].get_sosdisc_outputs().keys()),
                    ['residuals_history'])

            # elif isinstance(disc, SoSScatterData):
            #     self.assertListEqual(
            #         list(disc.get_data_io_dict('in').keys()), ['x_dict', 'scenario_list'])
            #     self.assertListEqual(
            #         list(disc.get_data_io_dict('out').keys()), ['scenario_A.x', 'scenario_B.x'])
            #     self.assertDictEqual(self.exec_eng.dm.get_value(
            #         f'{self.study_name}.multi_scenarios.x_dict'), {'scenario_A': 2, 'scenario_B': 4})
            #     self.assertEqual(self.exec_eng.dm.get_value(
            #         f'{self.study_name}.multi_scenarios.scenario_A.x'), 2)
            #     self.assertEqual(self.exec_eng.dm.get_value(
            #         f'{self.study_name}.multi_scenarios.scenario_B.x'), 4)

    def test_03_dump_and_load_after_execute(self):

        builders = self.exec_eng.factory.get_builder_from_process(
            repo=self.repo, mod_id='test_disc1_disc3_very_simple_multi_scenario')
        self.exec_eng.factory.set_builders_to_coupling_builder(
            builders)
        self.exec_eng.configure()

        dict_values = {}
        dict_values[f'{self.study_name}.multi_scenarios.builder_mode'] = 'multi_instance'
        dict_values[f'{self.study_name}.multi_scenarios.scenario_df'] = pd.DataFrame({'selected_scenario': [True,
                                                                                                            True],
                                                                                      'scenario_name': ['scenario_1',
                                                                                                        'scenario_2']})
        # dict_values[f'{self.study_name}.multi_scenarios.scenario_list'] = [
        #     'scenario_1', 'scenario_2']

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
                    '.multi_scenarios.name_list'] = ['name_1', 'name_2']
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_1.Disc3.z'] = 1.2
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_2.Disc3.z'] = 1.5
        dict_values[self.study_name + '.name_1.x'] = x1
        dict_values[self.study_name + '.name_2.x'] = x2

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        self.exec_eng.execute()

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_list'), ['scenario_1', 'scenario_2'])

        y1 = a1 * x1 + b1
        y2 = a2 * x2 + b2

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.name_1.y'), y1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.name_2.y'), y2)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.name_1.y'), y1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.name_2.y'), y2)

        dump_dir = join(self.root_dir, self.namespace)

        BaseStudyManager.static_dump_data(
            dump_dir, self.exec_eng, DirectLoadDump())

        exec_eng2 = ExecutionEngine(self.namespace)
        builders = exec_eng2.factory.get_builder_from_process(
            repo=self.repo, mod_id='test_disc1_disc3_very_simple_multi_scenario')
        exec_eng2.factory.set_builders_to_coupling_builder(builders)

        exec_eng2.configure()

        BaseStudyManager.static_load_data(
            dump_dir, exec_eng2, DirectLoadDump())

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.name_1.y'), y1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.name_2.y'), y2)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.name_1.y'), y1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.name_2.y'), y2)
        # Clean the dump folder at the end of the test
        self.dirs_to_del.append(
            join(self.root_dir, self.namespace))

    def _test_04_multi_scenario_of_scatter_parallel(self):
        if os.name == 'nt':
            print('INFO: Parallel execution of very simple ms deactivated on windows')
        else:
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
                'name_list', builder_list=builder_list, autogather=True)

            mod_list = f'{self.base_path}.disc3_scenario.Disc3'
            disc3_builder = self.exec_eng.factory.get_builder_from_module(
                'Disc3', mod_list)
            scatter_list.append(disc3_builder)

            multi_scenarios = self.exec_eng.factory.create_very_simple_multi_scenario_builder(
                'multi_scenarios', 'scenario_list', scatter_list, autogather=True, gather_node='Post-processing')

            self.exec_eng.factory.set_builders_to_coupling_builder(
                multi_scenarios)
            self.exec_eng.configure()

            dict_values = {}
            dict_values[f'{self.study_name}.multi_scenarios.builder_mode'] = 'multi_instance'
            dict_values[f'{self.study_name}.multi_scenarios.scenario_df'] = pd.DataFrame({'selected_scenario': [True,
                                                                                                                True],
                                                                                          'scenario_name': ['scenario_1',
                                                                                                            'scenario_2']})
            # dict_values[f'{self.study_name}.multi_scenarios.scenario_list'] = [
            #     'scenario_1', 'scenario_2']

            scenario_list = ['scenario_1', 'scenario_2']
            for scenario in scenario_list:
                x1 = 2.
                x2 = 4.
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
                        '.multi_scenarios.name_list'] = ['name_1', 'name_2']
            dict_values[self.study_name +
                        '.multi_scenarios.scenario_1.Disc3.z'] = 1.2
            dict_values[self.study_name +
                        '.multi_scenarios.scenario_2.Disc3.z'] = 1.5
            dict_values[self.study_name + '.name_1.x'] = x1
            dict_values[self.study_name + '.name_2.x'] = x2

            #- parallel option: set the number of subcouplings to run in parallel
            dict_values[self.study_name + '.n_subcouplings_parallel'] = 4

            self.exec_eng.load_study_from_input_dict(dict_values)
            self.exec_eng.display_treeview_nodes()

            self.exec_eng.execute()

            self.assertEqual(self.exec_eng.dm.get_value(
                'MyCase.multi_scenarios.scenario_list'), ['scenario_1', 'scenario_2'])

            y1 = a1 * x1 + b1
            y2 = a2 * x2 + b2

            self.assertEqual(self.exec_eng.dm.get_value(
                'MyCase.multi_scenarios.scenario_1.name_1.y'), y1)
            self.assertEqual(self.exec_eng.dm.get_value(
                'MyCase.multi_scenarios.scenario_1.name_2.y'), y2)
            self.assertEqual(self.exec_eng.dm.get_value(
                'MyCase.multi_scenarios.scenario_2.name_1.y'), y1)
            self.assertEqual(self.exec_eng.dm.get_value(
                'MyCase.multi_scenarios.scenario_2.name_2.y'), y2)

            gather_disc1 = self.exec_eng.dm.get_disciplines_with_name(
                'MyCase.Post-processing.Disc1')[0]
            self.assertListEqual(sorted([key for key in list(gather_disc1.get_data_in().keys()) if key not in gather_disc1.NUM_DESC_IN]), [
                'scenario_1.y_dict', 'scenario_2.y_dict', 'scenario_list'])
            self.assertListEqual(
                list(gather_disc1.get_data_out().keys()), ['y_dict'])
            self.assertDictEqual(gather_disc1.get_data_out()['y_dict']['value'], {
                                 'scenario_1.name_1': y1, 'scenario_1.name_2': y2, 'scenario_2.name_1': y1, 'scenario_2.name_2': y2})

            gather_disc3 = self.exec_eng.dm.get_disciplines_with_name(
                'MyCase.Post-processing.Disc3')[0]
            self.assertListEqual(sorted([key for key in list(gather_disc3.get_data_in().keys())if key not in ProxyDiscipline.NUM_DESC_IN]), [
                'scenario_1.o', 'scenario_2.o', 'scenario_list'])
            self.assertListEqual(
                list(gather_disc3.get_data_out().keys()), ['o_dict'])

    def test_05_get_samples_after_crash(self):
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
        self.exec_eng.ns_manager.add_ns(
            'ns_post_proc', 'MyCase.Post-processing')

        # instantiate factory # get instantiator from Discipline class

        builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                             mod_id='test_disc1_scenario')

        scatter_list = self.exec_eng.factory.create_multi_scatter_builder_from_list(
            'name_list', builder_list=builder_list, autogather=False)  # TODO: handle autogather input order and set to True...

        mod_list = f'{self.base_path}.disc3_scenario.Disc3'
        disc3_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc3', mod_list)
        scatter_list.append(disc3_builder)

        multi_scenarios = self.exec_eng.factory.create_very_simple_multi_scenario_driver(
            'multi_scenarios', 'scenario_list', scatter_list, autogather=True, gather_node='Post-processing')

        self.exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)
        self.exec_eng.configure()

        dict_values = {}
        dict_values[f'{self.study_name}.multi_scenarios.builder_mode'] = 'multi_instance'
        dict_values[f'{self.study_name}.multi_scenarios.scenario_df'] = pd.DataFrame({'selected_scenario': [True,
                                                                                                            True],
                                                                                      'scenario_name': ['scenario_1',
                                                                                                        'scenario_2']})
        # dict_values[f'{self.study_name}.multi_scenarios.scenario_list'] = [
        #     'scenario_1', 'scenario_2']

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
                    '.multi_scenarios.name_list'] = ['name_1', 'name_2']
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_1.Disc3.z'] = 1.2
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_2.Disc3.z'] = 1.5

        # missing input x1:
        # dict_values[self.study_name + '.name_1.x'] = x1

        dict_values[self.study_name + '.name_2.x'] = x2

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.dm.set_values_from_dict(dict_values)
        self.exec_eng.display_treeview_nodes()
        try:
             # execute fails because of missing input x1 => ValueError
            self.exec_eng.execute()
        except ValueError:
            # check that all variables have been loaded on dm at the end of the
            # execution
            for var in self.exec_eng.dm.data_id_map.keys():
                dm_value = self.exec_eng.dm.get_value(var)
                if var == f'{self.study_name}.multi_scenarios.scenario_list':
                    # this variable is an exception because it is forced by the value of another variable during setup
                    # TODO: remove this if when scatter as a tool is ready /!\
                    continue

                if var == f'{self.study_name}.multi_scenarios.scenario_df':
                    self.assertListEqual(
                        dict_values[var].values.tolist(), dm_value.values.tolist())
                    continue

                if var not in dict_values:
                    # default inputs and all outputs
                    self.assertEqual(self.exec_eng.dm.get_data(
                        var, 'default'), dm_value)
                else:
                    # user defined inputs
                    self.assertEqual(dict_values[var], dm_value)

    def test_06_scatter_node_namespace_removal_and_change_builder_mode_multi_to_mono(self):
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
        # NEED THE ns_eval TOO /!\
        self.exec_eng.ns_manager.add_ns(
            'ns_eval', 'MyCase.multi_scenarios')

        # instantiate factory # get instantiator from Discipline class
        builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                             mod_id='test_disc1_scenario')

        scatter_list = self.exec_eng.factory.create_multi_scatter_builder_from_list(
            'name_list', builder_list=builder_list, autogather=False)  # TODO: handle autogather input order and set to True...

        mod_path = f'{self.base_path}.disc3_scenario.Disc3'
        disc3_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc3', mod_path)
        scatter_list.append(disc3_builder)

        multi_scenarios = self.exec_eng.factory.create_very_simple_multi_scenario_driver(
            'multi_scenarios', 'scenario_list', scatter_list, autogather=True, gather_node='Post-processing')

        self.exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)
        self.exec_eng.configure()

        dict_values = {}
        dict_values[f'{self.study_name}.multi_scenarios.builder_mode'] = 'multi_instance'
        dict_values[f'{self.study_name}.multi_scenarios.scenario_df'] = pd.DataFrame({'selected_scenario': [True,
                                                                                                            True],
                                                                                      'scenario_name': ['scenario_1',
                                                                                                        'scenario_2']})
        dict_values[f'{self.study_name}.multi_scenarios.name_list'] = [
            'name_1', 'name_2']

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        # # check tree view with scenario_1 and scenario_2
        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       f'\t|_ multi_scenarios',
                       '\t\t|_ scenario_1',
                       '\t\t\t|_ Disc1',
                       '\t\t\t\t|_ name_1',
                       '\t\t\t\t|_ name_2',
                       '\t\t\t|_ Disc3',
                       '\t\t|_ scenario_2',
                       '\t\t\t|_ Disc1',
                       '\t\t\t\t|_ name_1',
                       '\t\t\t\t|_ name_2',
                       '\t\t\t|_ Disc3',
                       '\t|_ Post-processing',
                       # '\t\t|_ Disc1', # TODO: reactivate when activating gather role
                       '\t\t|_ Disc3',
                       '\t|_ name_1',
                       '\t|_ name_2']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        for disc in self.exec_eng.dm.get_disciplines_with_name('MyCase.multi_scenarios'):
            if isinstance(disc, ProxyDisciplineScatter):
                self.assertListEqual(list(disc.get_scattered_disciplines().keys()), [
                                     'scenario_1', 'scenario_2'])

        dict_values[f'{self.study_name}.multi_scenarios.builder_mode'] = 'mono_instance'
        # dict_values[self.study_name +
        #             '.multi_scenarios.scenario_list'] = ['scenario_1']

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        # check tree view after scenario_2 deletion to validate cleaning #TODO:
        # reactivate checks when treeview is fixed
        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       f'\t|_ multi_scenarios',
                       '\t\t|_ subprocess',
                       '\t\t\t|_ Disc1',
                       '\t\t\t\t|_ name_1',
                       '\t\t\t\t|_ name_2',
                       '\t\t\t|_ Disc3',
                       '\t|_ Post-processing',
                       # '\t\t|_ Disc1', # TODO: reactivate when activating gather role
                       '\t\t|_ Disc3',
                       '\t|_ name_1',
                       '\t|_ name_2']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def test_07_scatter_node_namespace_removal_and_change_builder_mode_mono_to_multi(self):
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
        # NEED THE ns_eval TOO /!\
        self.exec_eng.ns_manager.add_ns(
            'ns_eval', 'MyCase.multi_scenarios')

        # instantiate factory # get instantiator from Discipline class
        builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                             mod_id='test_disc1_scenario')

        scatter_list = self.exec_eng.factory.create_multi_scatter_builder_from_list(
            'name_list', builder_list=builder_list, autogather=False)  # TODO: handle autogather input order and set to True...

        mod_path = f'{self.base_path}.disc3_scenario.Disc3'
        disc3_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc3', mod_path)
        scatter_list.append(disc3_builder)

        multi_scenarios = self.exec_eng.factory.create_very_simple_multi_scenario_driver(
            'multi_scenarios', 'scenario_list', scatter_list, autogather=True, gather_node='Post-processing')

        self.exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)
        self.exec_eng.configure()

        dict_values = {}
        dict_values[f'{self.study_name}.multi_scenarios.builder_mode'] = 'mono_instance'
        dict_values[f'{self.study_name}.multi_scenarios.name_list'] = [
            'name_1', 'name_2']

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes(display_variables=True)

        # check tree view mono_instance #TODO: reactivate checks when treeview
        # is fixed
        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       f'\t|_ multi_scenarios',
                       '\t\t|_ subprocess',
                       '\t\t\t|_ Disc1',
                       '\t\t\t\t|_ name_1',
                       '\t\t\t\t|_ name_2',
                       '\t\t\t|_ Disc3',
                       '\t|_ Post-processing',
                       # '\t\t|_ Disc1', # TODO: reactivate when activating gather role
                       '\t\t|_ Disc3',
                       '\t|_ name_1',
                       '\t|_ name_2']
        exp_tv_str = '\n'.join(exp_tv_list)
        print(exp_tv_str)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        for disc in self.exec_eng.dm.get_disciplines_with_name('MyCase.multi_scenarios'):
            if isinstance(disc, ProxyDisciplineScatter):
                self.assertListEqual(list(disc.get_scattered_disciplines().keys()), [
                                     'scenario_1', 'scenario_2'])

        dict_values[f'{self.study_name}.multi_scenarios.builder_mode'] = 'multi_instance'
        dict_values[f'{self.study_name}.multi_scenarios.scenario_df'] = pd.DataFrame({'selected_scenario': [True,
                                                                                                            True],
                                                                                      'scenario_name': ['scenario_1',
                                                                                                        'scenario_2']})
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()
        # # check tree view with scenario_1 and scenario_2
        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       f'\t|_ multi_scenarios',
                       '\t\t|_ scenario_1',
                       '\t\t\t|_ Disc1',
                       '\t\t\t\t|_ name_1',
                       '\t\t\t\t|_ name_2',
                       '\t\t\t|_ Disc3',
                       '\t\t|_ scenario_2',
                       '\t\t\t|_ Disc1',
                       '\t\t\t\t|_ name_1',
                       '\t\t\t\t|_ name_2',
                       '\t\t\t|_ Disc3',
                       '\t|_ Post-processing',
                       # '\t\t|_ Disc1', # TODO: reactivate when activating gather role
                       '\t\t|_ Disc3',
                       '\t|_ name_1',
                       '\t|_ name_2']

        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()


if '__main__' == __name__:
    cls = TestVerySimpleMultiScenario()
    cls.setUp()
    cls.test_06_scatter_node_namespace_removal_and_change_builder_mode_multi_to_mono()
    cls.tearDown()
