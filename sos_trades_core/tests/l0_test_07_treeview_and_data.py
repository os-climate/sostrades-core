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
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from sos_trades_core.study_manager.base_study_manager import BaseStudyManager
from sos_trades_core.tools.tree.treenode import TreeNode
from sos_trades_core.tools.tree.treeview import TreeView


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
        self.repo = 'sos_trades_core.sos_processes.test'
        self.sub_proc = 'test_disc1_disc2_coupling'
        self.exec_eng = ExecutionEngine(self.namespace)
        self.factory = self.exec_eng.factory

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
            for disc in self.exec_eng.root_process.sos_disciplines:

                if disc.get_disc_full_name() == full_node_name:
                    data_io += list(disc.apply_visibility_ns('in')) + \
                        list(disc.apply_visibility_ns('out'))

            self.assertListEqual(sorted(list(set(data_io))),
                                 sorted(list(node.disc_data.keys())))

            for child_node in node.children:
                full_node_name = f'{my_case_node.name}.{node.name}.{child_node.name}'
                data_io = []
                for disc in self.exec_eng.root_process.sos_disciplines:

                    if disc.get_disc_full_name() == full_node_name:
                        data_io += list(disc.apply_visibility_ns('in')) + \
                            list(disc.apply_visibility_ns('out'))
                self.assertListEqual(sorted(list(set(data_io))),
                                     sorted(list(child_node.disc_data.keys())))

    def test_02_treeview_without_data(self):

        # Check that treeview is built without data and disc_data in nodes
        builders = self.factory.get_builder_from_process(
            repo=self.repo, mod_id='test_coupling_of_scatter')
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)

        self.exec_eng.configure()

        # User fill in the fields in the GUI
        dict_values = {self.study_name +
                       '.name_list': ['name_1', 'name_2']}
        self.exec_eng.dm.set_values_from_dict(dict_values)
        self.exec_eng.configure()

        self.exec_eng.get_treeview(True, False)

        my_case_node = self.exec_eng.dm.treeview.root

        self.assertListEqual([], list(my_case_node.data.keys()))

        self.assertListEqual([], list(
            my_case_node.disc_data.keys()))

        for node in my_case_node.children:
            self.assertListEqual([], list(node.disc_data.keys()))

            for child_node in node.children:
                self.assertListEqual([], list(child_node.disc_data.keys()))

    def test_03_treeview_read_only(self):

        # Check that treeview is built without data and disc_data in nodes
        builders = self.factory.get_builder_from_process(
            repo=self.repo, mod_id='test_coupling_of_scatter')
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)

        self.exec_eng.configure()

        # User fill in the fields in the GUI
        dict_values = {self.study_name +
                       '.name_list': ['name_1', 'name_2']}
        self.exec_eng.dm.set_values_from_dict(dict_values)
        self.exec_eng.configure()

        self.exec_eng.dm.create_treeview(
            self.exec_eng.root_process, False, True)

        my_case_node = self.exec_eng.dm.treeview.root

        # Test data are not editable
        for node_data in my_case_node.data.values():
            self.assertFalse(node_data[SoSDiscipline.EDITABLE])

        for node in my_case_node.children:
            for node_data in node.data.values():
                self.assertFalse(node_data[SoSDiscipline.EDITABLE])

            for child_node in node.children:
                for node_data in child_node.data.values():
                    self.assertFalse(node_data[SoSDiscipline.EDITABLE])

        # Test disc_data are not editable
        for node_data in my_case_node.disc_data.values():
            self.assertFalse(node_data[SoSDiscipline.EDITABLE])

        for node in my_case_node.children:
            for node_data in node.disc_data.values():
                self.assertFalse(node_data[SoSDiscipline.EDITABLE])

            for child_node in node.children:
                for node_data in child_node.disc_data.values():
                    self.assertFalse(node_data[SoSDiscipline.EDITABLE])

    def test_04_treeview_root_process_documentation(self):

        # retrieve treeview
        study_case_manager = BaseStudyManager(
            self.repo, self.sub_proc, self.study_name)

        # retrieve treeview
        treeview = study_case_manager.execution_engine.get_treeview()

        # check root process documentation
        root_process_documentation = treeview.root.markdown_documentation

        self.assertEqual(len(root_process_documentation), 1,
                         'Root node process documentation must contain one item')

        root_documentation = root_process_documentation[0]

        self.assertTrue(TreeNode.MARKDOWN_NAME_KEY in root_documentation,
                        f'Missing key {TreeNode.MARKDOWN_NAME_KEY} in markdown dict ')
        self.assertTrue(TreeNode.MARKDOWN_DOCUMENTATION_KEY in root_documentation,
                        f'Missing key {TreeNode.MARKDOWN_DOCUMENTATION_KEY} in markdown dict ')
        self.assertTrue(TreeView.PROCESS_DOCUMENTATION ==
                        root_documentation[TreeNode.MARKDOWN_NAME_KEY], f'Root markdown does not have the process documentation tag')
        self.assertTrue(len(
            root_documentation[TreeNode.MARKDOWN_DOCUMENTATION_KEY]) > 0, 'Markdown string must not be empty')

        self.assertEqual('# Test disc 1 disc 2 coupling',
                         root_documentation[TreeNode.MARKDOWN_DOCUMENTATION_KEY], 'Markdown string is not the intended one')
