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

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline


class TestDataManagerStorage(unittest.TestCase):
    """
    Class to test storage of data and disciplines in data manager
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.name = 'SoSDisc'
        self.ee = ExecutionEngine('Test')
        self.ns_test = 'Test'
        self.factory = self.ee.factory
        base_path = 'sostrades_core.sos_wrapping.test_discs'
        self.mod1_path = f'{base_path}.disc1.Disc1'
        self.mod2_path = f'{base_path}.disc2.Disc2'

    def test_01_data_dict(self):

        ns_dict = {'ns_ac': f'{self.ns_test}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = self.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        disc2_builder = self.factory.get_builder_from_module(
            'Disc2', self.mod2_path)

        self.factory.set_builders_to_coupling_builder(
            [disc1_builder, disc2_builder])

        self.ee.configure()

        # check data_dict and data_id_map lengths
        self.assertEqual(len(self.ee.dm.data_dict),
                         len(self.ee.dm.data_id_map))

        # check data id and full names
        for var_id in self.ee.dm.data_dict.keys():
            var_f_name = self.ee.dm.get_var_full_name(var_id)
            self.assertEqual(self.ee.dm.get_data_id(var_f_name), var_id)

        # check data_dict content
        self.assertIn('Test.Disc1.a', self.ee.dm.data_id_map.keys())
        self.assertIn('Test.y', self.ee.dm.data_id_map.keys())
        y_dependencies_id = self.ee.dm.get_data(
            'Test.y', ProxyDiscipline.DISCIPLINES_DEPENDENCIES)
        y_dependencies_names = [self.ee.dm.get_disc_full_name(disc_id)
                                for disc_id in y_dependencies_id]
        self.assertListEqual(y_dependencies_names, [
            'Test.Disc1', 'Test.Disc2'])

        disc_id_list = self.ee.dm.get_discipline_ids_list('Test.Disc1')
        # remove keys in DM
        self.ee.dm.remove_keys(disc_id_list[0], ['Test.Disc1.a', 'Test.y'], 'in')

        # check data_dict content after keys deletion
        self.assertNotIn('Test.Disc1.a', self.ee.dm.data_id_map.keys())
        self.assertIn('Test.y', self.ee.dm.data_id_map.keys())
        y_dependencies_id = self.ee.dm.get_data(
            'Test.y', ProxyDiscipline.DISCIPLINES_DEPENDENCIES)
        y_dependencies_names = [self.ee.dm.get_disc_full_name(disc_id)
                                for disc_id in y_dependencies_id]
        self.assertListEqual(y_dependencies_names, [
            'Test.Disc2'])

    def test_02_disciplines_dict(self):

        ns_dict = {'ns_ac': f'{self.ns_test}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = self.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        disc2_builder = self.factory.get_builder_from_module(
            'Disc2', self.mod2_path)

        self.factory.set_builders_to_coupling_builder(
            [disc1_builder, disc2_builder])

        self.ee.configure()

        # check disciplines_dict and disciplines_id_map lengths
        self.assertEqual(len(self.ee.dm.disciplines_dict),
                         len(self.ee.dm.disciplines_id_map))

        # check disciplines ids and full names
        for disc_id in self.ee.dm.disciplines_dict:
            disc_f_name = self.ee.dm.get_disc_full_name(disc_id)
            self.assertEqual(
                self.ee.dm.get_discipline_ids_list(disc_f_name), [disc_id])

        # check disciplines_dict content after keys deletion
        self.assertListEqual(list(self.ee.dm.disciplines_id_map.keys()), [
            'Test', 'Test.Disc1', 'Test.Disc2'])

        # remove Disc2
        disc2_id = self.ee.dm.get_discipline_ids_list('Test.Disc2')[0]
        self.ee.dm.clean_from_disc(disc2_id)

        self.assertRaises(
            KeyError, lambda: self.ee.dm.clean_from_disc(disc2_id))

        # check disciplines_dict and data_dict content after discipline
        # deletion
        self.assertListEqual(list(self.ee.dm.disciplines_id_map.keys()), [
            'Test', 'Test.Disc1'])

        self.assertNotIn('Test.Disc2.constant', self.ee.dm.data_id_map)
        self.assertNotIn('Test.Disc2.power', self.ee.dm.data_id_map)
        self.assertNotIn('Test.z', self.ee.dm.data_id_map)

        y_dependencies_id = self.ee.dm.get_data(
            'Test.y', ProxyDiscipline.DISCIPLINES_DEPENDENCIES)
        y_dependencies_names = [self.ee.dm.get_disc_full_name(disc_id)
                                for disc_id in y_dependencies_id]
        self.assertListEqual(y_dependencies_names, [
            'Test.Disc1'])

        # remove SoSCoupling Test
        disc1_id = self.ee.dm.get_discipline_ids_list('Test.Disc1')[0]
        self.ee.dm.clean_from_disc(disc1_id)

    def test_03_execute(self):

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

        self.assertEqual(self.ee.dm.get_value('Test.x'), None)
        self.ee.dm.set_data('Test.x', ProxyDiscipline.VALUE, 50.0)
        self.assertEqual(self.ee.dm.get_value('Test.x'), 50.0)

        a = 1.0
        b = 3.0
        x = 99.0
        values_dict = {self.ns_test + '.x': x,
                       self.ns_test + '.Disc1.a': a,
                       self.ns_test + '.Disc1.b': b,
                       self.ns_test + '.Disc2.constant': 1.5,
                       self.ns_test + '.Disc2.power': 2}

        self.ee.load_study_from_input_dict(values_dict)

        self.assertEqual(self.ee.dm.get_data('Test.x', 'value'), 99.0)
        self.assertEqual(self.ee.dm.get_value('Test.x'), 99.0)

        self.ee.execute()

    def test_04_namespace_change(self):

        ns_dict = {'ns_ac': f'{self.ns_test}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = self.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        disc2_builder = self.factory.get_builder_from_module(
            'Disc2', self.mod2_path)

        self.factory.set_builders_to_coupling_builder(
            [disc1_builder, disc2_builder])

        self.ee.configure()

        var_id = self.ee.dm.get_data_id('Test.x')
        self.assertEqual(self.ee.dm.data_id_map['Test.x'], var_id)

        ns_ac = self.ee.ns_manager.ns_list[0]
        ns_ac.update_value('New_ns_ac')
        self.ee.dm.generate_data_id_map()

        self.assertEqual(self.ee.dm.data_id_map['New_ns_ac.x'], var_id)
        self.assertNotIn('Test.y', self.ee.dm.data_id_map)
        self.assertIn('New_ns_ac.y', self.ee.dm.data_id_map)
        self.assertNotIn('Test.z', self.ee.dm.data_id_map)
        self.assertIn('New_ns_ac.z', self.ee.dm.data_id_map)

        test_id = self.ee.dm.disciplines_id_map['Test'][0]
        ns_test = self.ee.dm.disciplines_dict[test_id]['ns_reference']
        ns_test.update_value('New_ns_test')
        self.ee.dm.generate_disciplines_id_map()

        self.assertEqual(
            self.ee.dm.disciplines_id_map['New_ns_test'], [test_id])
        self.assertNotIn('Test', self.ee.dm.disciplines_id_map)

        self.assertIn('Test.sub_mda_class', self.ee.dm.data_id_map)
        self.ee.dm.generate_data_id_map()
        self.assertIn('New_ns_test.sub_mda_class', self.ee.dm.data_id_map)

    def test_05_convert_dict_with_maps(self):

        ns_dict = {'ns_ac': f'{self.ns_test}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = self.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        disc2_builder = self.factory.get_builder_from_module(
            'Disc2', self.mod2_path)

        self.factory.set_builders_to_coupling_builder(
            [disc1_builder, disc2_builder])

        self.ee.configure()

        self.assertDictEqual(self.ee.dm.data_dict,
                             self.ee.dm.convert_data_dict_with_ids(self.ee.dm.convert_data_dict_with_full_name()))

    def test_06_crash_with_distinct_disciplines_in_same_local_namespace_for_execution(self):
        same_name = 'SameName'
        ns_dict = {'ns_ac': f'{self.ns_test}'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = self.factory.get_builder_from_module(
            same_name, self.mod1_path)
        disc2_builder = self.factory.get_builder_from_module(
            same_name, self.mod2_path)

        self.factory.set_builders_to_coupling_builder(
            [disc1_builder, disc2_builder])

        with self.assertRaises(Exception) as cm:
            self.ee.configure()

        disc1_name = self.mod1_path.rsplit('.', 1)[0]
        disc2_name = self.mod2_path.rsplit('.', 1)[0]

        error_message = f'Trying to add two distinct disciplines with the same local namespace:' \
                        f' {self.ns_test}.{same_name} , classes are : {disc2_name} and {disc1_name}'
        # 'Trying to add two distinct disciplines with the same local namespace: '
        # 'Test.SameName , classes are : sostrades_core.sos_wrapping.test_discs.disc2 '
        # 'and sostrades_core.sos_wrapping.test_discs.disc1'
        self.assertEqual(str(cm.exception), error_message)

    def test_07_crash_with_distinct_disciplines_in_same_local_namespace_for_execution_from_usecase(self):
        builder_list = self.factory.get_builder_from_process(repo='sostrades_core.sos_processes.test',
                                                             mod_id='test_disc1_disc2_coupling_same_name')
        self.factory.set_builders_to_coupling_builder(builder_list)
        with self.assertRaises(Exception) as cm:
            self.ee.configure()
            pass
        same_name = 'SameName'
        disc1_name = self.mod1_path.rsplit('.', 1)[0]
        disc2_name = self.mod2_path.rsplit('.', 1)[0]
        error_message = f'Trying to add two distinct disciplines with the same local namespace:' \
                        f' {self.ns_test}.{same_name} , classes are : {disc2_name} and {disc1_name}'

        self.assertEqual(str(cm.exception), error_message)
