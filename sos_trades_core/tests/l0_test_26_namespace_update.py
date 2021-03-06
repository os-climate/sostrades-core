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


class TestNamespaceManagement(unittest.TestCase):
    """
    Class to test namespace management in processes
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.name = 'SoSDisc'
        self.ee = ExecutionEngine('Test')
        self.ns_test = 'Test'
        self.factory = self.ee.factory
        self.repo = 'sos_trades_core.sos_processes.test'
        base_path = 'sos_trades_core.sos_wrapping.test_discs'
        self.mod1_path = f'{base_path}.disc1.Disc1'
        self.mod2_path = f'{base_path}.disc2.Disc2'

    def test_01_get_builder_from_module(self):
        ns_dict = {'ns_ac': f'{self.ns_test}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = self.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        disc2_builder = self.factory.get_builder_from_module(
            'Disc2', self.mod2_path)

        self.factory.set_builders_to_coupling_builder(
            [disc1_builder, disc2_builder])

        self.ee.configure()
        self.ee.display_treeview_nodes()
        a = 1.0
        b = 3.0
        x = 99.0
        values_dict = {self.ns_test + '.x': x,
                       self.ns_test + '.Disc1.a': a,
                       self.ns_test + '.Disc1.b': b,
                       self.ns_test + '.Disc2.constant': 1.5,
                       self.ns_test + '.Disc2.power': 2}

        self.ee.dm.set_values_from_dict(values_dict)

        self.ee.execute()

        res = self.ee.dm.get_value(self.ns_test + '.y')

        self.assertEqual(res, a * x + b)
        self.assertEqual(self.ee.dm.get_value(
            self.ns_test + '.x'), values_dict[self.ns_test + '.x'])

    def test_02_change_order_disciplines(self):

        ns_dict = {'ns_ac': f'{self.ns_test}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = self.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        disc2_builder = self.factory.get_builder_from_module(
            'Disc2', self.mod2_path)

        self.factory.set_builders_to_coupling_builder(
            [disc1_builder, disc2_builder])

        self.ee.configure()
        self.ee.display_treeview_nodes()
        a = 1.0
        b = 3.0
        x = 99.0
        values_dict = {self.ns_test + '.x': x,
                       self.ns_test + '.Disc1.a': a,
                       self.ns_test + '.Disc1.b': b,
                       self.ns_test + '.Disc2.constant': 1.5,
                       self.ns_test + '.Disc2.power': 2}

        self.ee.dm.set_values_from_dict(values_dict)

        self.ee.execute()

        res = self.ee.dm.get_value(self.ns_test + '.y')

        self.assertEqual(res, a * x + b)
        self.assertEqual(self.ee.dm.get_value(
            self.ns_test + '.x'), values_dict[self.ns_test + '.x'])

    def test_03_add_discipline_with_process(self):

        ns_dict = {'ns_ac': f'{self.ns_test}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        builder = self.factory.get_builder_from_process(
            self.repo, 'test_disc1_disc2_coupling')

        self.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()
        a = 1.0
        b = 3.0
        x = 99.0
        values_dict = {self.ns_test + '.x': x,
                       self.ns_test + '.Disc1.a': a,
                       self.ns_test + '.Disc1.b': b,
                       self.ns_test + '.Disc2.constant': 1.5,
                       self.ns_test + '.Disc2.power': 2}

        for disc in self.ee.dm.disciplines_dict:
            print(disc)

        self.ee.dm.set_values_from_dict(values_dict)

        self.ee.execute()

        res = self.ee.dm.get_value(self.ns_test + '.y')

        self.assertEqual(res, a * x + b)
        self.assertEqual(self.ee.dm.get_value(
            self.ns_test + '.x'), values_dict[self.ns_test + '.x'])

    def test_04_check_dependency_list(self):

        ns_dict = {'ns_ac': f'{self.ns_test}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = self.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        disc2_builder = self.factory.get_builder_from_module(
            'Disc2', self.mod2_path)

        self.factory.set_builders_to_coupling_builder(
            [disc1_builder, disc2_builder])

        self.ee.configure()
        ns_ac_disc_list = self.ee.ns_manager.shared_ns_dict['ns_ac'].get_dependency_disc_list(
        )
        self.assertListEqual(
            [f'{self.ns_test}.Disc1', f'{self.ns_test}.Disc2'], [self.ee.dm.get_disc_full_name(disc_id) for disc_id in ns_ac_disc_list])

        self.ee.configure()
        disc1_id = ns_ac_disc_list[0]
        disc1 = self.ee.dm.get_discipline(disc1_id)
        self.factory.clean_discipline_list([disc1])

        self.assertListEqual(
            [None, f'{self.ns_test}.Disc2'], [self.ee.dm.get_disc_full_name(disc_id) for disc_id in ns_ac_disc_list])

        disc2_id = ns_ac_disc_list[1]
        disc2 = self.ee.dm.get_discipline(disc2_id)
        self.factory.clean_discipline_list([disc2])

        self.assertListEqual(
            [None, None], [self.ee.dm.get_disc_full_name(disc_id) for disc_id in ns_ac_disc_list])

        self.ee.ns_manager.clean_ns_without_dependencies()

        self.assertListEqual(
            [], [ns.name for ns in self.ee.ns_manager.ns_list])

    def test_05_update_shared_namespace_with_extra_ns(self):

        ns_dict = {'ns_ac': f'{self.ns_test}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = self.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        disc2_builder = self.factory.get_builder_from_module(
            'Disc2', self.mod2_path)

        self.factory.set_builders_to_coupling_builder(
            [disc1_builder, disc2_builder])

        self.ee.configure()

        a = 1.0
        b = 3.0
        x = 99.0
        values_dict = {self.ns_test + '.x': x,
                       self.ns_test + '.Disc1.a': a,
                       self.ns_test + '.Disc1.b': b,
                       self.ns_test + '.Disc2.constant': 1.5,
                       self.ns_test + '.Disc2.power': 2}

        self.ee.dm.set_values_from_dict(values_dict)

        # Now that the complete use case is set we change the local namespace
        self.ee.ns_manager.update_all_shared_namespaces_by_name(
            'extraNS', 'ns_ac')
        self.ee.configure()
        self.ee.display_treeview_nodes()

        self.assertListEqual(['Test', 'Test.Disc1', 'Test.Disc2'], list(
            self.ee.dm.disciplines_id_map.keys()))

        self.ee.execute()

        res = self.ee.dm.get_value('extraNS.' + self.ns_test + '.y')

        self.assertEqual(res, a * x + b)
        self.assertEqual(self.ee.dm.get_value('extraNS.' +
                                              self.ns_test + '.x'), values_dict[self.ns_test + '.x'])

    def test_06_update_shared_namespaces_and_builders_with_extra_name(self):

        ns_dict = {'ns_ac': f'{self.ns_test}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = self.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        disc2_builder = self.factory.get_builder_from_module(
            'Disc2', self.mod2_path)

        extra_name = 'extra_name'
        # update namespace list with extra_ns
        self.ee.ns_manager.update_namespace_list_with_extra_ns(
            extra_name, after_name=self.ee.study_name)
        # update builder names with extra_name
        self.ee.factory.update_builder_list_with_extra_name(
            extra_name, [disc1_builder, disc2_builder])

        self.factory.set_builders_to_coupling_builder(
            [disc1_builder, disc2_builder])

        self.ee.configure()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns_test}',
                       f'|_ {self.ns_test}',
                       f'\t|_ extra_name',
                       '\t\t|_ Disc1',
                       '\t\t|_ Disc2']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.ee.display_treeview_nodes()

        a = 1.0
        b = 3.0
        x = 99.0
        values_dict = {self.ns_test + '.extra_name.x': x,
                       self.ns_test + '.extra_name.Disc1.a': a,
                       self.ns_test + '.extra_name.Disc1.b': b,
                       self.ns_test + '.extra_name.Disc2.constant': 1.5,
                       self.ns_test + '.extra_name.Disc2.power': 2}

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()
