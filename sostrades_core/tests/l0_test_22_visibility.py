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


class TestVisibility(unittest.TestCase):
    """
    Visibility test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)
        self.factory = self.ee.factory
        self.repo = 'sostrades_core.sos_processes.test'
        base_path = 'sostrades_core.sos_wrapping.test_discs'
        self.mod1_path = f'{base_path}.disc1.Disc1'
        self.mod2_path = f'{base_path}.disc2.Disc2'
        self.mod1_path_internal = f'{base_path}.disc1_internal.Disc1'

    def test_01_check_local_visibility(self):

        ns_dict = {'ns_ac': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)
        disc1_name = 'Disc1'
        disc1_builder = self.factory.get_builder_from_module(
            disc1_name, self.mod1_path)
        disc2_name = 'Disc2'
        disc2_builder = self.factory.get_builder_from_module(
            disc2_name, self.mod2_path)

        self.factory.set_builders_to_coupling_builder(
            [disc1_builder, disc2_builder])

        self.ee.configure()
        self.ee.display_treeview_nodes()

        dm = self.ee.dm

        ref_local_ns = {'a': f'{self.name}.{disc1_name}.a',
                        'b': f'{self.name}.{disc1_name}.b',
                        'indicator': f'{self.name}.{disc1_name}.indicator',
                        'constant': f'{self.name}.{disc2_name}.constant',
                        'power': f'{self.name}.{disc2_name}.power'}
        for key_id, dm_dict in dm.data_dict.items():
            full_name = dm.get_var_full_name(key_id)
            var_name = dm_dict[ProxyDiscipline.VAR_NAME]
            ns_value = dm_dict[ProxyDiscipline.NS_REFERENCE].get_value()
            editable = dm_dict[ProxyDiscipline.EDITABLE]
            io_type = dm_dict[ProxyDiscipline.IO_TYPE]
            if var_name in ref_local_ns:
                # check if the full_name is correct
                self.assertEqual(full_name, ref_local_ns[var_name])
                # Check if the namespace_value in the namespace object is
                # correct
                self.assertEqual(
                    ref_local_ns[var_name], f'{ns_value}.{var_name}')
                if io_type == ProxyDiscipline.IO_TYPE_IN:
                    self.assertTrue(editable)
                else:
                    self.assertFalse(editable)

    def test_02_check_shared_visibility(self):

        ns_dict = {'ns_ac': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)
        disc1_name = 'Disc1'
        disc1_builder = self.factory.get_builder_from_module(
            disc1_name, self.mod1_path)
        disc2_name = 'Disc2'
        disc2_builder = self.factory.get_builder_from_module(
            disc2_name, self.mod2_path)

        self.factory.set_builders_to_coupling_builder(
            [disc1_builder, disc2_builder])

        self.ee.configure()
        self.ee.display_treeview_nodes()

        dm = self.ee.dm
        ns_shared_ref = ns_dict['ns_ac']
        ref_shared_ns = {'x': f'{ns_shared_ref}.x',
                         'y': f'{ns_shared_ref}.y',
                         'z': f'{ns_shared_ref}.z', }
        for key_id, dm_dict in dm.data_dict.items():
            full_name = dm.get_var_full_name(key_id)
            var_name = dm_dict[ProxyDiscipline.VAR_NAME]
            ns_value = dm_dict[ProxyDiscipline.NS_REFERENCE].get_value()
            if var_name in ref_shared_ns:
                # check if the full_name is correct
                self.assertEqual(full_name, ref_shared_ns[var_name])
                # Check if the namespace_value in the namespace object is
                # correct
                self.assertEqual(
                    ref_shared_ns[var_name], f'{ns_value}.{var_name}')

    def test_03_check_internal_visibility(self):

        ns_dict = {'ns_ac': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)
        disc1_name = 'Disc1_internal'
        disc1_builder = self.factory.get_builder_from_module(
            disc1_name, self.mod1_path_internal)
        disc2_name = 'Disc2'
        disc2_builder = self.factory.get_builder_from_module(
            disc2_name, self.mod2_path)

        self.factory.set_builders_to_coupling_builder(
            [disc1_builder, disc2_builder])

        self.ee.configure()
        self.ee.display_treeview_nodes()

        dm = self.ee.dm

        ref_local_ns = {'a': f'{self.name}.{disc1_name}.a'}
        for key_id, dm_dict in dm.data_dict.items():
            full_name = dm.get_var_full_name(key_id)
            var_name = dm_dict[ProxyDiscipline.VAR_NAME]
            ns_value = dm_dict[ProxyDiscipline.NS_REFERENCE].get_value()
            editable = dm_dict[ProxyDiscipline.EDITABLE]
            if var_name in ref_local_ns:
                # check if the full_name is correct
                self.assertEqual(full_name, ref_local_ns[var_name])
                # Check if the namespace_value in the namespace object is
                # correct
                self.assertEqual(
                    ref_local_ns[var_name], f'{ns_value}.{var_name}')

                self.assertFalse(editable)

    def test_04_execute_with_internal_visibility(self):

        ns_dict = {'ns_ac': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)
        disc1_name = 'Disc1_internal'
        disc1_builder = self.factory.get_builder_from_module(
            disc1_name, self.mod1_path_internal)
        disc2_name = 'Disc2'
        disc2_builder = self.factory.get_builder_from_module(
            disc2_name, self.mod2_path)

        self.factory.set_builders_to_coupling_builder(
            [disc1_builder, disc2_builder])

        self.ee.configure()
        self.ee.display_treeview_nodes()

        values_dict = {}
        values_dict[f'{self.name}.Disc1_internal.b'] = 20.
        values_dict[f'{self.name}.Disc2.power'] = 2
        values_dict[f'{self.name}.Disc2.constant'] = -10.
        values_dict[f'{self.name}.x'] = 3.

        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()
