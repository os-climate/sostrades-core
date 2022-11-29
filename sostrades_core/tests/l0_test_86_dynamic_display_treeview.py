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
from sostrades_core.execution_engine.proxy_driver_evaluator import ProxyDriverEvaluator
'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import unittest
from logging import Handler

from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class TestConfigDependencyDiscs(unittest.TestCase):
    """
    Tool building test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.dirs_to_del = []
        self.study_name = 'MyCase'
        self.exec_eng = ExecutionEngine(self.study_name)
        self.factory = self.exec_eng.factory

        self.repo = 'sostrades_core.sos_processes.test'

    def test_01_display_existing_variable_ns(self):

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc1.Disc1'
        disc1_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc1', mod_list)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc2.Disc2'
        disc2_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc2', mod_list)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            [disc1_builder, disc2_builder])

        display_value = f'{self.exec_eng.study_name}.new_display'
        self.exec_eng.ns_manager.add_ns(
            'ns_ac', self.exec_eng.study_name, display_value=display_value)

        ns_ac = self.exec_eng.ns_manager.get_ns_in_shared_ns_dict('ns_ac')

        self.assertEqual(ns_ac.get_display_value(), display_value)
        self.exec_eng.configure()

        self.exec_eng.display_treeview_nodes()
        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       f'\t|_ Disc1',
                       f'\t|_ Disc2',
                       f'\t|_ new_display', ]

        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        treeview_display_var = self.exec_eng.display_treeview_nodes(
            display_variables=True)

        assert '\n\t\t-> x' == treeview_display_var.split('new_display')[-1]

        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       f'\t|_ Disc1',
                       f'\t|_ Disc2', ]

        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes(
            exec_display=True)

    def test_02_display_existing_disc_ns(self):

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc1.Disc1'
        disc1_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc1', mod_list)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc2.Disc2'
        disc2_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc2', mod_list)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            [disc1_builder, disc2_builder])

        self.exec_eng.ns_manager.add_ns(
            'ns_ac', self.exec_eng.study_name)
        self.exec_eng.ns_manager.add_display_ns_to_disc(
            disc1_builder, f'{self.study_name}.New_ns_disc')

        self.exec_eng.configure()

        self.exec_eng.display_treeview_nodes(display_variables=True)
        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       f'\t|_ New_ns_disc',
                       f'\t|_ Disc2']

        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        self.exec_eng.ns_manager.add_display_ns_to_disc(
            disc1_builder, f'{self.study_name}.New_ns_disc_new')
        self.exec_eng.dm.treeview = None
        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       f'\t|_ New_ns_disc_new',
                       f'\t|_ Disc2']

        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        self.exec_eng.ns_manager.add_display_ns_to_disc(
            disc2_builder, f'{self.study_name}.New_ns_disc2')
        self.exec_eng.dm.treeview = None
        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       f'\t|_ New_ns_disc_new',
                       f'\t|_ New_ns_disc2']

        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        self.exec_eng.ns_manager.add_display_ns_to_disc(
            disc1_builder, f'{self.study_name}')
        self.exec_eng.ns_manager.add_display_ns_to_disc(
            disc2_builder, f'{self.study_name}')
        self.exec_eng.dm.treeview = None

        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}']

        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       f'\t|_ Disc1',
                       f'\t|_ Disc2']

        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes(
            exec_display=True)


if '__main__' == __name__:
    cls = TestConfigDependencyDiscs()
    cls.setUp()
    cls.test_02_display_existing_disc_ns()
    cls.tearDown()
