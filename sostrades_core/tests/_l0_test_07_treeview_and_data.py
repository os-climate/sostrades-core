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
from sostrades_core.tools.post_processing.post_processing_factory import PostProcessingFactory
'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import unittest

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.execution_engine.discipline_proxy import DisciplineProxy
from sostrades_core.study_manager.base_study_manager import BaseStudyManager
from sostrades_core.tools.tree.treenode import TreeNode
from sostrades_core.tools.tree.treeview import TreeView


class TestTreeviewAndData(unittest.TestCase):
    """
    Treeview and data test class
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
        self.base_path = 'sostrades_core.sos_wrapping.test_discs'

    def test_01_treeview_scatter_gather(self):

        # load process in GUI
        builders = self.factory.get_builder_from_process(
            repo=self.repo, mod_id='test_coupling_of_scatter')
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)

        self.exec_eng.configure()

        # User fill in the fields in the GUI
        dict_values = {self.study_name +
                       '.name_list': ['name_1', 'name_2']}
        self.exec_eng.dm.set_values_from_dict(dict_values)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes(display_variables=True)

        my_case_node = self.exec_eng.dm.treeview.root

        study_name_data = [key for key in self.exec_eng.dm.data_id_map.keys() if (
            len(key.split('.')) == 2 and key.split('.')[0] == self.namespace)]

        self.assertListEqual(study_name_data, list(my_case_node.data.keys()))
        my_case_data_io = list(self.exec_eng.root_process.apply_visibility_ns(
            'in')) + list(self.exec_eng.root_process.apply_visibility_ns('out'))

        self.assertListEqual(my_case_data_io, list(
            my_case_node.disc_data.keys()))

        for node in my_case_node.children:
            data_io = []
            full_node_name = f'{my_case_node.name}.{node.name}'
            for disc in self.exec_eng.root_process.proxy_disciplines:

                if disc.get_disc_full_name() == full_node_name:
                    data_io += list(disc.apply_visibility_ns('in')) + \
                        list(disc.apply_visibility_ns('out'))

            self.assertListEqual(sorted(list(set(data_io))),
                                 sorted(list(node.disc_data.keys())))

            for child_node in node.children:
                full_node_name = f'{my_case_node.name}.{node.name}.{child_node.name}'
                data_io = []
                for disc in self.exec_eng.root_process.proxy_disciplines:

                    if disc.get_disc_full_name() == full_node_name:
                        data_io += list(disc.apply_visibility_ns('in')) + \
                            list(disc.apply_visibility_ns('out'))
                self.assertListEqual(sorted(list(set(data_io))),
                                     sorted(list(child_node.disc_data.keys())))
    #
    # def test_02_treeview_without_data(self):
    #
    #     # Check that treeview is built without data and disc_data in nodes
    #     builders = self.factory.get_builder_from_process(
    #         repo=self.repo, mod_id='test_coupling_of_scatter')
    #     self.exec_eng.factory.set_builders_to_coupling_builder(builders)
    #
    #     self.exec_eng.configure()
    #
    #     # User fill in the fields in the GUI
    #     dict_values = {self.study_name +
    #                    '.name_list': ['name_1', 'name_2']}
    #     self.exec_eng.dm.set_values_from_dict(dict_values)
    #     self.exec_eng.configure()
    #
    #     self.exec_eng.get_treeview(True, False)
    #
    #     my_case_node = self.exec_eng.dm.treeview.root
    #
    #     self.assertListEqual([], list(my_case_node.data.keys()))
    #
    #     self.assertListEqual([], list(
    #         my_case_node.disc_data.keys()))
    #
    #     for node in my_case_node.children:
    #         self.assertListEqual([], list(node.disc_data.keys()))
    #
    #         for child_node in node.children:
    #             self.assertListEqual([], list(child_node.disc_data.keys()))
    #
    # def test_03_treeview_read_only(self):
    #
    #     # Check that treeview is built without data and disc_data in nodes
    #     builders = self.factory.get_builder_from_process(
    #         repo=self.repo, mod_id='test_coupling_of_scatter')
    #     self.exec_eng.factory.set_builders_to_coupling_builder(builders)
    #
    #     self.exec_eng.configure()
    #
    #     # User fill in the fields in the GUI
    #     dict_values = {self.study_name +
    #                    '.name_list': ['name_1', 'name_2']}
    #     self.exec_eng.dm.set_values_from_dict(dict_values)
    #     self.exec_eng.configure()
    #
    #     self.exec_eng.dm.create_treeview(
    #         self.exec_eng.root_process, False, True)
    #
    #     my_case_node = self.exec_eng.dm.treeview.root
    #
    #     # Test data are not editable
    #     for node_data in my_case_node.data.values():
    #         self.assertFalse(node_data[SoSDiscipline.EDITABLE])
    #
    #     for node in my_case_node.children:
    #         for node_data in node.data.values():
    #             self.assertFalse(node_data[SoSDiscipline.EDITABLE])
    #
    #         for child_node in node.children:
    #             for node_data in child_node.data.values():
    #                 self.assertFalse(node_data[SoSDiscipline.EDITABLE])
    #
    #     # Test disc_data are not editable
    #     for node_data in my_case_node.disc_data.values():
    #         self.assertFalse(node_data[SoSDiscipline.EDITABLE])
    #
    #     for node in my_case_node.children:
    #         for node_data in node.disc_data.values():
    #             self.assertFalse(node_data[SoSDiscipline.EDITABLE])
    #
    #         for child_node in node.children:
    #             for node_data in child_node.disc_data.values():
    #                 self.assertFalse(node_data[SoSDiscipline.EDITABLE])
    #
    # def test_04_treeview_root_process_documentation(self):
    #
    #     # retrieve treeview
    #     study_case_manager = BaseStudyManager(
    #         self.repo, self.sub_proc, self.study_name)
    #
    #     # retrieve treeview
    #     treeview = study_case_manager.execution_engine.get_treeview()
    #
    #     # check root process documentation
    #     root_process_documentation = treeview.root.markdown_documentation
    #
    #     self.assertEqual(len(root_process_documentation), 1,
    #                      'Root node process documentation must contain one item')
    #
    #     root_documentation = root_process_documentation[0]
    #
    #     self.assertTrue(TreeNode.MARKDOWN_NAME_KEY in root_documentation,
    #                     f'Missing key {TreeNode.MARKDOWN_NAME_KEY} in markdown dict ')
    #     self.assertTrue(TreeNode.MARKDOWN_DOCUMENTATION_KEY in root_documentation,
    #                     f'Missing key {TreeNode.MARKDOWN_DOCUMENTATION_KEY} in markdown dict ')
    #     self.assertTrue(TreeView.PROCESS_DOCUMENTATION ==
    #                     root_documentation[TreeNode.MARKDOWN_NAME_KEY], f'Root markdown does not have the process documentation tag')
    #     self.assertTrue(len(
    #         root_documentation[TreeNode.MARKDOWN_DOCUMENTATION_KEY]) > 0, 'Markdown string must not be empty')
    #
    #     self.assertEqual('# Test disc 1 disc 2 coupling',
    #                      root_documentation[TreeNode.MARKDOWN_DOCUMENTATION_KEY], 'Markdown string is not the intended one')

    # def test_05_treeview_multiscenario_postproc_node_without_data(self):
    #
    #     # Check that post processing node is visible when thereis no data on the node
    #             # scatter build map
    #     ac_map = {'input_name': 'name_list',
    #               'input_type': 'string_list',
    #               'input_ns': 'ns_scatter_scenario',
    #               'output_name': 'ac_name',
    #               'scatter_ns': 'ns_ac',
    #               'gather_ns': 'ns_scenario',
    #               'ns_to_update': ['ns_data_ac']}
    #
    #     self.exec_eng.smaps_manager.add_build_map('name_list', ac_map)
    #
    #     # scenario build map
    #     scenario_map = {'input_name': 'scenario_list',
    #                     'input_type': 'string_list',
    #                     'input_ns': 'ns_scatter_scenario',
    #                     'output_name': 'scenario_name',
    #                     'scatter_ns': 'ns_scenario',
    #                     'gather_ns': 'ns_scatter_scenario',
    #                     'ns_to_update': ['ns_disc3', 'ns_barrierr', 'ns_out_disc3']}
    #
    #     self.exec_eng.smaps_manager.add_build_map(
    #         'scenario_list', scenario_map)
    #
    #     # shared namespace
    #     self.exec_eng.ns_manager.add_ns('ns_barrierr', 'MyCase')
    #     self.exec_eng.ns_manager.add_ns(
    #         'ns_scatter_scenario', 'MyCase.multi_scenarios')
    #     self.exec_eng.ns_manager.add_ns(
    #         'ns_disc3', 'MyCase.multi_scenarios.Disc3')
    #     self.exec_eng.ns_manager.add_ns(
    #         'ns_out_disc3', 'MyCase.multi_scenarios')
    #     self.exec_eng.ns_manager.add_ns(
    #         'ns_data_ac', 'MyCase')
    #     self.exec_eng.ns_manager.add_ns(
    #         'ns_post_proc', 'MyCase.Post-processing')
    #
    #     # instantiate factory # get instantiator from Discipline class
    #
    #     builder_list = self.factory.get_builder_from_process(repo=self.repo,
    #                                                          mod_id='test_disc1_scenario')
    #
    #     scatter_list = self.exec_eng.factory.create_multi_scatter_builder_from_list(
    #         'name_list', builder_list=builder_list, autogather=True)
    #
    #     mod_list = f'{self.base_path}.disc3_scenario.Disc3'
    #     disc3_builder = self.exec_eng.factory.get_builder_from_module(
    #         'Disc3', mod_list)
    #     scatter_list.append(disc3_builder)
    #
    #     multi_scenarios = self.exec_eng.factory.create_very_simple_multi_scenario_builder(
    #         'multi_scenarios', 'scenario_list', scatter_list, autogather=False)
    #
    #     # add post-processing on 'Post-processing' node by loading a module
    #     # with implemented graphs
    #     self.exec_eng.post_processing_manager.add_post_processing_module_to_namespace(
    #         'ns_post_proc', 'sostrades_core.sos_wrapping.test_discs.chart_post_proc_multi_scenario')
    #
    #     self.exec_eng.factory.set_builders_to_coupling_builder(
    #         multi_scenarios)
    #     self.exec_eng.configure()
    #
    #     dict_values = {}
    #     dict_values[f'{self.study_name}.multi_scenarios.scenario_list'] = [
    #         'scenario_1', 'scenario_2']
    #
    #     self.exec_eng.load_study_from_input_dict(dict_values)
    #     self.exec_eng.display_treeview_nodes()
    #
    #     scenario_list = ['scenario_1', 'scenario_2']
    #     for scenario in scenario_list:
    #         x1 = 2.
    #         x2 = 4.
    #         a1 = 3
    #         b1 = 4
    #         a2 = 6
    #         b2 = 2
    #
    #         dict_values[self.study_name + '.name_1.a'] = a1
    #         dict_values[self.study_name + '.name_2.a'] = a2
    #         dict_values[self.study_name + '.multi_scenarios.' +
    #                     scenario + '.Disc1.name_1.b'] = b1
    #         dict_values[self.study_name + '.multi_scenarios.' +
    #                     scenario + '.Disc1.name_2.b'] = b2
    #         dict_values[self.study_name + '.multi_scenarios.' +
    #                     scenario + '.Disc3.constant'] = 3
    #         dict_values[self.study_name + '.multi_scenarios.' +
    #                     scenario + '.Disc3.power'] = 2
    #
    #     dict_values[self.study_name +
    #                 '.multi_scenarios.name_list'] = ['name_1', 'name_2']
    #     dict_values[self.study_name +
    #                 '.multi_scenarios.scenario_1.Disc3.z'] = 1.2
    #     dict_values[self.study_name +
    #                 '.multi_scenarios.scenario_2.Disc3.z'] = 1.5
    #     dict_values[self.study_name + '.name_1.x'] = x1
    #     dict_values[self.study_name + '.name_2.x'] = x2
    #
    #     self.exec_eng.load_study_from_input_dict(dict_values)
    #     self.exec_eng.display_treeview_nodes()
    #
    #     self.exec_eng.execute()
    #     my_case_node = self.exec_eng.dm.treeview.root
    #     self.assertTrue(
    #         'Post-processing' in [child.name for child in my_case_node.children])
    #
    #     ppf = PostProcessingFactory()
    #     filters = ppf.get_post_processing_filters_by_namespace(
    #         self.exec_eng, f'{self.study_name}.Post-processing')
    #     graph_list = ppf.get_post_processing_by_namespace(self.exec_eng, f'{self.study_name}.Post-processing',
    #                                                       filters, as_json=False)
    #
    #     self.assertTrue(len(graph_list) == 1)


if '__main__' == __name__:
    cls = TestTreeviewAndData()
    cls.setUp()
    cls.test_05_treeview_multiscenario_postproc_node_without_data()
